"""Write the small database's lookups as a packed list: the same IDs and ages as agerating-ids.db, about a fifth of
its size gzipped, for apps that read it with nothing but a gzip decoder.

Layout (integers little-endian, varints LEB128: 7 bits per byte, low bits first, high bit set on all but the last):
  "AGID"                  magic
  u8   1                  format version
  u8   N                  number of tables, then per table:
    u8   len, len bytes   table name (ASCII): imdb, tmdb_movie, tmdb_tv, tvdb_series, tvdb_movie
    u32  count            rows
    count varints         the IDs in ascending order, each as its gap from the previous one (the first from 0)
    count u8              the ages, in the same order (0, 6, 12, 16 or 18)
The tables, IDs and ages are those of agerating-ids.db (build_ids_db.py), which documents what each table holds.
Usage: python pack_ids.py [IDS_DB] [OUT]   (default data/agerating-ids.db -> data/agerating-ids.bin)
"""
import argparse, os, sqlite3
from pathlib import Path
from build_ids_db import TABLES

MAGIC, VERSION = b"AGID", 1


def varint(n, out):
    while n > 0x7F:
        out.append(n & 0x7F | 0x80)
        n >>= 7
    out.append(n)


def pack(ids_db, out):
    con = sqlite3.connect(f"file:{ids_db}?mode=ro", uri=True)
    data = bytearray(MAGIC)
    data += bytes([VERSION, len(TABLES)])
    for table in TABLES:
        rows = con.execute(f"SELECT id, age FROM {table} ORDER BY id").fetchall()
        name = table.encode("ascii")
        data += bytes([len(name)]) + name + len(rows).to_bytes(4, "little")
        previous = 0
        for id_, _ in rows:
            if id_ < previous:
                raise SystemExit(f"{table}: id {id_} is negative or out of order")
            varint(id_ - previous, data)
            previous = id_
        data += bytes(age for _, age in rows)
    con.close()
    tmp = out.with_suffix(".packing")
    tmp.write_bytes(data)
    os.replace(tmp, out)
    print(f"packed {out} ({len(data) / 2**20:.1f} MB)", flush=True)


def read(path):
    """The packed list as {table: {id: age}}: the reference reader, used to check a file against its database."""
    data = Path(path).read_bytes()
    if data[:4] != MAGIC or data[4] != VERSION:
        raise ValueError(f"{path}: not a version {VERSION} packed list")
    pos, tables = 6, {}
    for _ in range(data[5]):
        name = data[pos + 1:pos + 1 + data[pos]].decode("ascii")
        pos += 1 + data[pos]
        count = int.from_bytes(data[pos:pos + 4], "little")
        pos += 4
        ids, previous = [], 0
        for _ in range(count):
            gap = shift = 0
            while True:
                byte = data[pos]
                pos += 1
                gap |= (byte & 0x7F) << shift
                shift += 7
                if byte < 0x80:
                    break
            previous += gap
            ids.append(previous)
        tables[name] = dict(zip(ids, data[pos:pos + count]))
        pos += count
    if pos != len(data):
        raise ValueError(f"{path}: {len(data) - pos} bytes after the last table")
    return tables


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ids_db", nargs="?", type=Path, default=Path("data/agerating-ids.db"))
    p.add_argument("out", nargs="?", type=Path, default=Path("data/agerating-ids.bin"))
    p.add_argument("--check", action="store_true", help="read the written file back and compare it with IDS_DB")
    a = p.parse_args()
    pack(a.ids_db, a.out)
    if a.check:
        con = sqlite3.connect(f"file:{a.ids_db}?mode=ro", uri=True)
        expected = {t: dict(con.execute(f"SELECT id, age FROM {t}")) for t in TABLES}
        if read(a.out) != expected:
            raise SystemExit(f"{a.out} doesn't read back as {a.ids_db}")
        print(f"checked: {sum(map(len, expected.values())):,} rows read back the same", flush=True)
