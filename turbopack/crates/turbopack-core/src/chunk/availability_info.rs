use anyhow::Result;
use bincode::{Decode, Encode};
use bitfield::bitfield;
use turbo_rcstr::RcStr;
use turbo_tasks::{ResolvedVc, Vc};

use crate::{
    chunk::available_chunk_groups::AvailableChunkGroups,
    module::{Module, Modules},
    module_graph::{
        ModuleGraph,
        chunk_group_info::{
            AsyncLoaderRelevantAvailabilities, ChunkGroupKey, RoaringBitmapWrapper,
        },
        module_batch::ChunkableModuleOrBatch,
    },
};

bitfield! {
    #[turbo_tasks::task_input]
    #[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Encode, Decode)]
    pub struct AvailabilityFlags(u8);
    impl Debug;
    pub is_in_async_module, set_is_in_async_module: 0;
}

#[turbo_tasks::task_input]
#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug, Encode, Decode)]
pub struct AvailabilityInfo {
    flags: AvailabilityFlags,
    /// Chunk groups whose contents are already available.
    available_chunk_groups: Option<ResolvedVc<AvailableChunkGroups>>,
    /// The root ChunkGroup::Entry
    entry_group: Option<ResolvedVc<Modules>>,
}

impl AvailabilityInfo {
    pub fn root() -> Self {
        Self {
            flags: AvailabilityFlags::default(),
            available_chunk_groups: None,
            entry_group: None,
        }
    }

    pub fn available_chunk_groups(&self) -> Option<ResolvedVc<AvailableChunkGroups>> {
        self.available_chunk_groups
    }

    pub async fn with_chunk_group(
        self,
        module_graph: ResolvedVc<ModuleGraph>,
        chunk_group: ChunkGroupKey,
    ) -> Result<Self> {
        let chunk_group = module_graph
            .chunk_group_info()
            .get_index_of(chunk_group)
            .await? as u32;
        Ok(Self {
            flags: self.flags,
            available_chunk_groups: Some(
                if let Some(available_chunk_groups) = self.available_chunk_groups {
                    available_chunk_groups
                        .with_chunk_group(*module_graph, chunk_group)
                        .to_resolved()
                        .await?
                } else {
                    AvailableChunkGroups::new(*module_graph, chunk_group)
                        .to_resolved()
                        .await?
                },
            ),
            entry_group: self.entry_group,
        })
    }

    /// Whether the module or module batch is already part of one of the available chunk groups.
    pub async fn is_available(
        &self,
        module_graph: ResolvedVc<ModuleGraph>,
        item: ChunkableModuleOrBatch,
    ) -> Result<bool> {
        let Some(available_chunk_groups) = self.available_chunk_groups else {
            return Ok(false);
        };
        let available_chunk_groups = available_chunk_groups.await?;
        let chunk_group_info = module_graph.chunk_group_info().await?;
        let module = match item {
            ChunkableModuleOrBatch::Module(module) => ResolvedVc::upcast(module),
            // All modules of a batch have the same chunk groups, so the first one is
            // representative.
            ChunkableModuleOrBatch::Batch(batch) => match batch.await?.modules.first() {
                Some(&module) => ResolvedVc::upcast(module),
                None => return Ok(false),
            },
            ChunkableModuleOrBatch::None(_) => return Ok(false),
        };
        Ok(chunk_group_info
            .module_chunk_groups
            .await?
            .get(&module)
            .is_some_and(|groups| {
                !groups.is_disjoint(available_chunk_groups.chunk_groups(module_graph))
            }))
    }

    pub fn in_async_module(self) -> Self {
        let mut flags = self.flags;
        flags.set_is_in_async_module(true);
        Self {
            flags,
            available_chunk_groups: self.available_chunk_groups,
            entry_group: self.entry_group,
        }
    }

    pub fn is_in_async_module(&self) -> bool {
        self.flags.is_in_async_module()
    }

    pub fn with_entry_group(self, entry_group: ResolvedVc<Modules>) -> Self {
        Self {
            flags: self.flags,
            available_chunk_groups: self.available_chunk_groups,
            entry_group: Some(entry_group),
        }
    }

    pub fn entry_group(&self) -> Option<ResolvedVc<Modules>> {
        self.entry_group
    }

    /// Replaces this availability with a canonical one for chunking the async chunk group of
    /// `module` (and everything chunked from it). Use it for the availability an async loader is
    /// created with.
    ///
    /// The result chunks the same as `self`, because it gives the same answer to every
    /// availability test that chunking makes (see
    /// [`crate::module_graph::chunk_group_info::AsyncLoaderRelevantAvailability::canonicalize`]).
    /// It drops the entry group when nothing reads it. Every parent chunk group that makes the
    /// same modules available therefore gets the same result. Without this, each parent would
    /// create its own async loader for `module`, because its availability always contains
    /// itself.
    ///
    /// The canonical available chunk groups may include chunk groups that are not in `self`.
    /// The narrowed chunk groups are created through [`AvailableChunkGroups::from_chunk_groups`],
    /// so equal results share one cell. This is intentionally not a `turbo_tasks::function`: a
    /// task per parent availability and module is the overhead this avoids. It reads a single
    /// entry of the keyed [`AsyncLoaderRelevantAvailabilities`] cell.
    ///
    /// A `module` without an async chunk group in `module_graph` keeps the availability as is.
    pub async fn for_async_loader(
        self,
        module_graph: ResolvedVc<ModuleGraph>,
        module: ResolvedVc<Box<dyn Module>>,
    ) -> Result<Self> {
        self.narrow_for_async_loader(
            module_graph,
            module_graph
                .chunk_group_info()
                .async_loader_relevant_availability(),
            module,
        )
        .await
    }

    /// [`AvailabilityInfo::for_async_loader`] with the relevant availabilities of `module_graph`
    /// passed in.
    pub(crate) async fn narrow_for_async_loader(
        self,
        module_graph: ResolvedVc<ModuleGraph>,
        relevant_availabilities: Vc<AsyncLoaderRelevantAvailabilities>,
        module: ResolvedVc<Box<dyn Module>>,
    ) -> Result<Self> {
        let Some(relevant) = relevant_availabilities.get(&module).await? else {
            return Ok(self);
        };
        let available_chunk_groups = match self.available_chunk_groups {
            Some(available_chunk_groups) => {
                let available_chunk_groups = available_chunk_groups.await?;
                let canonical =
                    relevant.canonicalize(available_chunk_groups.chunk_groups(module_graph));
                if canonical.is_empty() {
                    None
                } else {
                    Some(
                        AvailableChunkGroups::from_chunk_groups(
                            *module_graph,
                            RoaringBitmapWrapper(canonical),
                        )
                        .to_resolved()
                        .await?,
                    )
                }
            }
            None => None,
        };
        Ok(Self {
            flags: self.flags,
            available_chunk_groups,
            entry_group: if relevant.uses_entry_group {
                self.entry_group
            } else {
                None
            },
        })
    }

    pub async fn ident(&self) -> Result<Option<RcStr>> {
        Ok(
            if let Some(available_chunk_groups) = self.available_chunk_groups {
                Some(available_chunk_groups.hash().await?.to_string().into())
            } else {
                None
            },
        )
    }
}
