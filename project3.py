#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple


#   file  and B-Tree shape
BLOCK_SIZE = 512
MAGIC = b"4348PRJ3"
MIN_DEGREE = 10
MAX_KEYS = 2 * MIN_DEGREE - 1
MAX_CHILDREN = MAX_KEYS + 1
UINT64_MAX = 2**64 - 1


class IndexOops(Exception):
    """Lightweight signal that the on-disk structure isn't what we hoped."""


@dataclass
class FileBadge:
    """Tiny struct mirroring the header block on disk."""
    root_slot: int
    next_slot: int


@dataclass
class Blocky:
    """In-memory view of a single B-Tree node block."""
    block_id: int
    parent: int
    num_keys: int
    keys: List[int]
    values: List[int]
    children: List[int]
    dirty: bool = False

    def looks_leafy(self) -> bool:
        
        return all(kid == 0 for kid in self.children[: self.num_keys + 1])


def pack8(value: int) -> bytes:
    """Coerce an int into the 8-byte big-endian format the file expects."""
    return int(value).to_bytes(8, "big", signed=False)


def unpack8(blob: bytes) -> int:
    """Reverse of pack8: turn 8 bytes back into an int."""
    return int.from_bytes(blob, "big", signed=False)


class DiskLedger:
    """All the low-level file IO lives here: headers, nodes, allocations."""
    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.header: FileBadge | None = None

    def __enter__(self) -> DiskLedger:
        if not self.path.exists():
            raise FileNotFoundError(f"Index file {self.path} not found")
        self.handle = self.path.open("r+b")
        self._pull_header()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None

    @classmethod
    def create(cls, path: Path) -> None:
        # creating file with an empty header block.
        if path.exists():
            raise FileExistsError(f"File {path} already exists")
        blob = bytearray(BLOCK_SIZE)
        blob[:8] = MAGIC
        blob[8:16] = pack8(0)
        blob[16:24] = pack8(1)
        with path.open("xb") as handle:
            handle.write(blob)
            handle.flush()

    def _need_handle(self) -> None:
        if self.handle is None:
            raise RuntimeError("Index file is not open")

    def _pull_header(self) -> None:
        #Load the 512 byte head  
        self._need_handle()
        self.handle.seek(0)
        data = self.handle.read(BLOCK_SIZE)
        if len(data) != BLOCK_SIZE:
            raise IndexOops("Unable to read header block")
        if data[:8] != MAGIC:
            raise IndexOops("Invalid magic number")
        self.header = FileBadge(
            root_slot=unpack8(data[8:16]),
            next_slot=unpack8(data[16:24]),
        )

    def store_header(self) -> None:
        # adding header structure we have back out to disk
        self._need_handle()
        if self.header is None:
            raise RuntimeError("Header not loaded")
        blob = bytearray(BLOCK_SIZE)
        blob[:8] = MAGIC
        blob[8:16] = pack8(self.header.root_slot)
        blob[16:24] = pack8(self.header.next_slot)
        self.handle.seek(0)
        self.handle.write(blob)
        self.handle.flush()

    def read_block(self, block_id: int) -> Blocky:
        #Read a  node block and turn it into a Blocky
        self._need_handle()
        offset = block_id * BLOCK_SIZE
        self.handle.seek(offset)
        data = self.handle.read(BLOCK_SIZE)
        if len(data) != BLOCK_SIZE:
            raise IndexOops(f"Unable to read block {block_id}")
        stored_id = unpack8(data[0:8])
        if stored_id != block_id:
            raise IndexOops(f"Block id mismatch for {block_id}")
        parent = unpack8(data[8:16])
        num_keys = unpack8(data[16:24])
        if num_keys > MAX_KEYS:
            raise IndexOops(f"Node {block_id} reports too many keys")
        keys = [unpack8(data[24 + i * 8 : 24 + (i + 1) * 8]) for i in range(MAX_KEYS)]
        values_offset = 24 + MAX_KEYS * 8
        values = [
            unpack8(data[values_offset + i * 8 : values_offset + (i + 1) * 8])
            for i in range(MAX_KEYS)
        ]
        children_offset = values_offset + MAX_KEYS * 8
        children = [
            unpack8(data[children_offset + i * 8 : children_offset + (i + 1) * 8])
            for i in range(MAX_CHILDREN)
        ]
        return Blocky(
            block_id=block_id,
            parent=parent,
            num_keys=num_keys,
            keys=keys,
            values=values,
            children=children,
            dirty=False,
        )

    def write_block(self, block: Blocky) -> None:
        #serialize a  into its fixed-width on-disk 
        self._need_handle()
        blob = bytearray(BLOCK_SIZE)
        blob[0:8] = pack8(block.block_id)
        blob[8:16] = pack8(block.parent)
        blob[16:24] = pack8(block.num_keys)
        for i in range(MAX_KEYS):
            start = 24 + i * 8
            blob[start : start + 8] = pack8(block.keys[i])
        values_offset = 24 + MAX_KEYS * 8
        for i in range(MAX_KEYS):
            start = values_offset + i * 8
            blob[start : start + 8] = pack8(block.values[i])
        children_offset = values_offset + MAX_KEYS * 8
        for i in range(MAX_CHILDREN):
            start = children_offset + i * 8
            blob[start : start + 8] = pack8(block.children[i])
        offset = block.block_id * BLOCK_SIZE
        self.handle.seek(offset)
        self.handle.write(blob)
        self.handle.flush()
        block.dirty = False

    def mint_block(self, parent_id: int) -> Blocky:
        #adding a -new block id and return a blank node for it
        if self.header is None:
            raise RuntimeError("Header not loaded")
        new_id = self.header.next_slot
        self.header.next_slot += 1
        self.store_header()
        return Blocky(
            block_id=new_id,
            parent=parent_id,
            num_keys=0,
            keys=[0] * MAX_KEYS,
            values=[0] * MAX_KEYS,
            children=[0] * MAX_CHILDREN,
            dirty=True,
        )


class BlockGate:
    # resource manager  for 3-node residency limi
    def __init__(self, disk: DiskLedger):
        self.disk = disk
        self.active = 0
        self.limit = 3  # spec says no more than 3 nodes in memory

    def _grab_slot(self) -> None:
        #Count each  node so we dont go over limit
        if self.active >= self.limit:
            raise RuntimeError("Exceeded node memory limit (3)")
        self.active += 1

    def _drop_slot(self) -> None:
        self.active -= 1

    @contextmanager
    def load(self, block_id: int):
        # manager that loads a block yields it and flushes if dirty
        self._grab_slot()
        block: Blocky | None = None
        try:
            block = self.disk.read_block(block_id)
            yield block
        finally:
            if block is not None and block.dirty:
                self.disk.write_block(block)
            self._drop_slot()

    @contextmanager
    def new(self, parent_id: int):
        #Same idea as load but for new nodes
        self._grab_slot()
        block: Blocky | None = None
        try:
            block = self.disk.mint_block(parent_id)
            yield block
        finally:
            if block is not None:
                self.disk.write_block(block)
            self._drop_slot()


class TreeDriver:
    #higher-level B-Tree logic glued on top of the disk-backed nod
    def __init__(self, path: Path):
        self.disk = DiskLedger(path)
        self.blocks = BlockGate(self.disk)

    def __enter__(self) -> TreeDriver:
        self.disk.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disk.__exit__(exc_type, exc, tb)

    def insert(self, key: int, value: int) -> None:
        #public insert entry point sets up the root and delegates the rest
        header = self.disk.header
        if header is None:
            raise RuntimeError("Header not loaded")
        if header.root_slot == 0:
            with self.blocks.new(parent_id=0) as root:
                root.keys[0] = key
                root.values[0] = value
                root.num_keys = 1
                root.dirty = True
                root_block = root.block_id
            header.root_slot = root_block
            self.disk.store_header()
            return

        with self.blocks.load(header.root_slot) as root:
            if root.num_keys == MAX_KEYS:
                with self.blocks.new(parent_id=0) as fresh_root:
                    fresh_root.children[0] = root.block_id
                    fresh_root.num_keys = 0
                    fresh_root.dirty = True
                    root.parent = fresh_root.block_id
                    root.dirty = True
                    self._split_child(fresh_root, 0, root)
                    header.root_slot = fresh_root.block_id
                self.disk.store_header()
                self._insert_nonfull(header.root_slot, key, value)
                return
        self._insert_nonfull(header.root_slot, key, value)

    def _insert_nonfull(self, block_id: int, key: int, value: int) -> None:
        # B-Tree insert loop 
        current = block_id
        while True:
            to_reparent: List[Tuple[int, int]] = []
            with self.blocks.load(current) as node:
                for idx in range(node.num_keys):
                    if node.keys[idx] == key:
                        node.values[idx] = value
                        node.dirty = True
                        return
                if node.looks_leafy():
                    self._drop_into_leaf(node, key, value)
                    return
                child_idx = node.num_keys - 1
                while child_idx >= 0 and key < node.keys[child_idx]:
                    child_idx -= 1
                child_idx += 1
                child_id = node.children[child_idx]
                if child_id == 0:
                    raise IndexOops("Encountered missing child pointer")
                with self.blocks.load(child_id) as child:
                    if child.num_keys == MAX_KEYS:
                        new_child_id, moved = self._split_child(node, child_idx, child)
                        for kid in moved:
                            if kid != 0:
                                to_reparent.append((kid, new_child_id))
                        if key > node.keys[child_idx]:
                            child_id = node.children[child_idx + 1]
                next_hop = child_id
            self._fix_parents(to_reparent)
            current = next_hop

    def _drop_into_leaf(self, node: Blocky, key: int, value: int) -> None:
        #Slide keys rightward inside a leaf  add the new pair into place
        idx = node.num_keys - 1
        while idx >= 0 and node.keys[idx] > key:
            node.keys[idx + 1] = node.keys[idx]
            node.values[idx + 1] = node.values[idx]
            idx -= 1
        node.keys[idx + 1] = key
        node.values[idx + 1] = value
        node.num_keys += 1
        node.dirty = True





    def _split_child(self, parent: Blocky, index: int, child: Blocky) -> Tuple[int, List[int]]:
        #split a full child node and push the median key up to the parent
        t = MIN_DEGREE
        mid_key = child.keys[t - 1]
        mid_val = child.values[t - 1]
        moved_children: List[int] = []
        with self.blocks.new(parent.block_id) as newbie:
            newbie.parent = parent.block_id
            newbie.num_keys = t - 1
            for j in range(t - 1):
                newbie.keys[j] = child.keys[j + t]
                newbie.values[j] = child.values[j + t]
                child.keys[j + t] = 0
                child.values[j + t] = 0
            if not child.looks_leafy():
                for j in range(t):
                    moved_id = child.children[j + t]
                    newbie.children[j] = moved_id
                    moved_children.append(moved_id)
                    child.children[j + t] = 0
            else:
                moved_children = []
            child.num_keys = t - 1
            child.keys[t - 1] = 0
            child.values[t - 1] = 0
            child.dirty = True
            for j in range(parent.num_keys, index, -1):
                parent.children[j + 1] = parent.children[j]
            parent.children[index + 1] = newbie.block_id
            for j in range(parent.num_keys - 1, index - 1, -1):
                parent.keys[j + 1] = parent.keys[j]
                parent.values[j + 1] = parent.values[j]
            parent.keys[index] = mid_key
            parent.values[index] = mid_val
            parent.num_keys += 1
            parent.dirty = True
            newbie_id = newbie.block_id
        return newbie_id, moved_children

    def _fix_parents(self, pairs: Sequence[Tuple[int, int]]) -> None:
        #Update parent pointers for any kids that moved during a split
        for child_id, parent_id in pairs:
            with self.blocks.load(child_id) as node:
                node.parent = parent_id
                node.dirty = True

    def search(self, key: int) -> Tuple[int, int] | None:
        # search that walks down the tree until a key hits or misses
        header = self.disk.header
        if header is None or header.root_slot == 0:
            return None
        current = header.root_slot
        while current != 0:
            with self.blocks.load(current) as node:
                idx = 0
                while idx < node.num_keys and key > node.keys[idx]:
                    idx += 1
                if idx < node.num_keys and key == node.keys[idx]:
                    return node.keys[idx], node.values[idx]
                current = node.children[idx] if idx <= node.num_keys else 0
        return None

    def iter_pairs(self) -> Iterator[Tuple[int, int]]:
        #Inorder traversal using  stack 
        header = self.disk.header
        if header is None or header.root_slot == 0:
            return
        stack: List[Tuple[str, int, int]] = [("node", header.root_slot, 0)]
        while stack:
            action, ident, extra = stack.pop()
            if action == "emit":
                yield ident, extra
                continue
            block_id = ident
            with self.blocks.load(block_id) as node:
                if node.children[node.num_keys] != 0:
                    stack.append(("node", node.children[node.num_keys], 0))
                for idx in range(node.num_keys - 1, -1, -1):
                    stack.append(("emit", node.keys[idx], node.values[idx]))
                    if node.children[idx] != 0:
                        stack.append(("node", node.children[idx], 0))


def grab_uintish(value: str) -> int:
    #parse an integer string (supports 0x hex) and enforce uint64 bounds
    try:
        number = int(value, 0)
    except ValueError as exc:
        raise ValueError(f"Invalid integer: {value}") from exc
    if number < 0 or number > UINT64_MAX:
        raise ValueError(f"Value {value} is outside uint64 range")
    return number


def handle_create(args: List[str]) -> None:
    """CLI plumbing for the create command."""
    if len(args) != 1:
        complain_and_exit("create requires exactly one argument (index file)")
    path = Path(args[0])
    try:
        DiskLedger.create(path)
    except FileExistsError:
        complain_and_exit(f"File {path} already exists")
    except OSError as exc:
        complain_and_exit(f"Unable to create file {path}: {exc}")


def handle_insert(args: List[str]) -> None:
    """CLI plumbing for inserting a single key/value pair."""
    if len(args) != 3:
        complain_and_exit("insert requires: indexfile key value")
    path = Path(args[0])
    try:
        key = grab_uintish(args[1])
        value = grab_uintish(args[2])
    except ValueError as exc:
        complain_and_exit(str(exc))
    try:
        with TreeDriver(path) as tree:
            tree.insert(key, value)
    except (IndexOops, FileNotFoundError, RuntimeError) as exc:
        complain_and_exit(str(exc))


def handle_search(args: List[str]) -> None:
    # search
    if len(args) != 2:
        complain_and_exit("search requires: indexfile key")
    path = Path(args[0])
    try:
        key = grab_uintish(args[1])
    except ValueError as exc:
        complain_and_exit(str(exc))
    try:
        with TreeDriver(path) as tree:
            result = tree.search(key)
    except (IndexOops, FileNotFoundError, RuntimeError) as exc:
        complain_and_exit(str(exc))
    if result is None:
        complain_and_exit("Key not found")
    print(f"{result[0]} {result[1]}")


def handle_print(args: List[str]) -> None:
    # printing every key/value pair
    if len(args) != 1:
        complain_and_exit("print requires: indexfile")
    path = Path(args[0])
    try:
        with TreeDriver(path) as tree:
            for key, value in tree.iter_pairs():
                print(f"{key} {value}")
    except (IndexOops, FileNotFoundError, RuntimeError) as exc:
        complain_and_exit(str(exc))


def handle_extract(args: List[str]) -> None:
    # plumbing for CSV export
    if len(args) != 2:
        complain_and_exit("extract requires: indexfile output.csv")
    index_path = Path(args[0])
    output_path = Path(args[1])
    if output_path.exists():
        complain_and_exit(f"Output file {output_path} already exists")
    try:
        with TreeDriver(index_path) as tree:
            with output_path.open("x", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for key, value in tree.iter_pairs():
                    writer.writerow([key, value])
    except (IndexOops, FileNotFoundError, RuntimeError, OSError) as exc:
        complain_and_exit(str(exc))


def handle_load(args: List[str]) -> None:
    # plumbing for CSV import
    if len(args) != 2:
        complain_and_exit("load requires: indexfile input.csv")
    index_path = Path(args[0])
    csv_path = Path(args[1])
    if not csv_path.exists():
        complain_and_exit(f"CSV file {csv_path} not found")
    try:
        with TreeDriver(index_path) as tree:
            for key, value in _read_csvish(csv_path):
                tree.insert(key, value)
    except (IndexOops, FileNotFoundError, RuntimeError, ValueError) as exc:
        complain_and_exit(str(exc))


def _read_csvish(path: Path) -> Iterator[Tuple[int, int]]:
    # (key, value) tuples from the provided CSV file
    with path.open("r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for lineno, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"Line {lineno}: expected key,value")
            key = grab_uintish(row[0].strip())
            value = grab_uintish(row[1].strip())
            yield key, value


def complain_and_exit(message: str) -> None:
    # stderr + exit helper so messaging stays consistent
    print(message, file=sys.stderr)
    sys.exit(1)


# basic jump table so verbs map to their handler functions
COMMANDS = {
    "create": handle_create,
    "insert": handle_insert,
    "search": handle_search,
    "print": handle_print,
    "extract": handle_extract,
    "load": handle_load,
}


def main(argv: List[str]) -> None:
    """Parse the verb and punt to whichever handler matches."""
    if len(argv) < 2:
        complain_and_exit("usage: project3 <command> [args]")
    verb = argv[1].lower().strip()
    handler = COMMANDS.get(verb)
    if handler is None:
        complain_and_exit(f"Unknown command: {verb}")
    handler(argv[2:])


if __name__ == "__main__":
    main(sys.argv)

