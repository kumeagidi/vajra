from sarathi.core.block_space_manager.base_block_space_manager import (
    BaseBlockSpaceManager,
    BlockTable,
)
from sarathi.core.datatypes.sequence import Sequence


class MnemosyneBlockSpaceManager(BaseBlockSpaceManager):
    def allocate(self, seq: Sequence, num_blocks: int) -> None:
        # Allocate new physical token blocks that will store the prompt tokens.
        block_table: BlockTable = []
        for _ in range(num_blocks):
            block = self.gpu_allocator.allocate()
            block_table.append(block)

        self.block_tables[seq.seq_id] = block_table

    def allocate_delta(self, seq: Sequence, total_num_blocks: int) -> None:
        # Allocate new physical token blocks that will store the prompt tokens.
        if seq.seq_id not in self.block_tables:
            self.allocate(seq, total_num_blocks)
            return

        num_existing_blocks = len(self.block_tables[seq.seq_id])
        num_new_blocks = total_num_blocks - num_existing_blocks
        for _ in range(num_new_blocks):
            block = self.gpu_allocator.allocate()
            self.block_tables[seq.seq_id].append(block)

    def append_slot(self, seq: Sequence, num_total_blocks: int) -> bool:
        """
        Allocate a physical slot for a new token.
        It returns True if a new block is allocated.
        """
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if num_total_blocks < len(logical_blocks):
            # The sequence has a new logical block.
            # Allocate a new physical block.
            block = self.gpu_allocator.allocate()
            block_table.append(block)
            return True

        return False

    def get_num_initial_blocks(self, seq: Sequence) -> int:
        raise RuntimeError(
            "The mnemosyne scheduler is responsible for determining the blocks."
        )
