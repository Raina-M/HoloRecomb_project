"""Read chromosome names and lengths (default) or the Juicer statistics block (--stats) from the header of a .hic file
(formats v8 and v9). Usage: python hic_header.py file.hic [--stats]"""
import struct, sys

def read_header(path):
    f = open(path, 'rb')
    assert f.read(4) == b'HIC\x00', 'not a .hic file'
    version = struct.unpack('<i', f.read(4))[0]; f.read(8)                  # version, footer position
    def s():
        b = b''
        while (c := f.read(1)) != b'\0': b += c
        return b.decode(errors='replace')
    s()                                                                      # genome id
    if version >= 9: f.read(16)                                              # normalisation vector index position and length
    attrs = {s(): s() for _ in range(struct.unpack('<i', f.read(4))[0])}
    chroms = []
    for _ in range(struct.unpack('<i', f.read(4))[0]):
        name = s(); L = struct.unpack('<q' if version >= 9 else '<i', f.read(8 if version >= 9 else 4))[0]; chroms.append((name, L))
    return version, attrs, chroms

if __name__ == '__main__':
    v, attrs, chroms = read_header(sys.argv[1])
    if '--stats' in sys.argv: print(attrs.get('statistics', ''))
    else:
        for n, L in chroms:
            if n.lower() != 'all': print(n, L)
