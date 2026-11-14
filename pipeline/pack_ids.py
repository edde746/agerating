"""Write the small database's lookups as two packed lists, for apps that read them with nothing but a gzip decoder:

  agerating-ids.trave.gz  a Trave file (https://github.com/edde746/trave), magic "AGRT", gzipped. One table per ID type
                          (imdb, tmdb_movie, tmdb_tv, tvdb_series, tvdb_movie), each with the columns id (int_delta,
                          ascending) and age (int: 0, 6, 12, 16 or 18).
  agerating-ids.bin       the legacy AGID layout, kept for Tortuga's released versions until they read the Trave file
                          (weekly.sh gzips it). Integers little-endian, varints LEB128 (7 bits per byte, low bits
                          first, high bit set on all but the last):
    "AGID"                  magic
    u8   1                  format version
    u8   N                  number of tables, then per table:
      u8   len, len bytes   table name (ASCII), as above
      u32  count            rows
      count varints         the IDs in ascending order, each as its gap from the previous one (the first from 0)
      count u8              the ages, in the same order
The tables, IDs and ages are those of agerating-ids.db (build_ids_db.py), which documents what each table holds.
Usage: python pack_ids.py [IDS_DB] [OUT_DIR] [--check]   (default data/agerating-ids.db -> data/)
"""
import argparse, os, sqlite3
from pathlib import Path
import trave
from build_ids_db import TABLES

MAGIC = b"AGRT"
LEGACY_MAGIC, LEGACY_VERSION = b"AGID", 1


def load(ids_db):
    """{table: [(id, age), ...]} in ascending id order."""
    con = sqlite3.connect(f"file:{ids_db}?mode=ro", uri=True)
    tables = {t: con.execute(f"SELECT id, age FROM {t} ORDER BY id").fetchall() for t in TABLES}
    con.close()
    for table, rows in tables.items():
        if rows and rows[0][0] < 0:
            raise SystemExit(f"{table}: id {rows[0][0]} is negative")
    return tables


def write(path, data):
    tmp = path.with_name(path.name + ".packing")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    print(f"packed {path} ({len(data) / 2**20:.1f} MB)", flush=True)


def pack(tables):
    """The Trave file, gzipped."""
    return trave.gzip_bytes(trave.pack(MAGIC, [
        (t, len(rows), [("id", trave.INT_DELTA, [i for i, _ in rows]), ("age", trave.INT, [a for _, a in rows])])
        for t, rows in tables.items()]))


# Legacy AGID writer and reader: kept for Tortuga's released versions until they read agerating-ids.trave.gz.
def varint(n, out):
    while n > 0x7F:
        out.append(n & 0x7F | 0x80)
        n >>= 7
    out.append(n)


def pack_legacy(tables):
    data = bytearray(LEGACY_MAGIC)
    data += bytes([LEGACY_VERSION, len(tables)])
    for table, rows in tables.items():
        name = table.encode("ascii")
        data += bytes([len(name)]) + name + len(rows).to_bytes(4, "little")
        previous = 0
        for id_, _ in rows:
            varint(id_ - previous, data)
            previous = id_
        data += bytes(age for _, age in rows)
    return bytes(data)


def read_legacy(path):
    """The legacy packed list as {table: {id: age}}: the reference reader, used to check a file against its database."""
    data = Path(path).read_bytes()
    if data[:4] != LEGACY_MAGIC or data[4] != LEGACY_VERSION:
        raise ValueError(f"{path}: not a version {LEGACY_VERSION} AGID packed list")
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


def read(path):
    """The Trave file as {table: {id: age}}."""
    return {t: dict(zip(cols["id"], cols["age"])) for t, (_, cols) in trave.read(Path(path).read_bytes(), MAGIC).items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ids_db", nargs="?", type=Path, default=Path("data/agerating-ids.db"))
    p.add_argument("out_dir", nargs="?", type=Path, default=Path("data"))
    p.add_argument("--check", action="store_true", help="read both written files back and compare them with IDS_DB")
    a = p.parse_args()
    tables = load(a.ids_db)
    legacy, current = a.out_dir / "agerating-ids.bin", a.out_dir / "agerating-ids.trave.gz"
    write(legacy, pack_legacy(tables))
    write(current, pack(tables))
    if a.check:
        expected = {t: dict(rows) for t, rows in tables.items()}
        for path, reader in ((legacy, read_legacy), (current, read)):
            if reader(path) != expected:
                raise SystemExit(f"{path} doesn't read back as {a.ids_db}")
        print(f"checked: {sum(map(len, expected.values())):,} rows read back the same from both files", flush=True)
