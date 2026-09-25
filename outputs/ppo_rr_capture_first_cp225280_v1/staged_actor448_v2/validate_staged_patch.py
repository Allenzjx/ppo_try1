"""Read/AST-check staged textual edits without writing or importing candidates."""
import ast
import hashlib
from pathlib import Path

DIRECTORY=Path(__file__).resolve().parent
ROOT=DIRECTORY.parents[2]


def candidate(patch_path):
    lines=patch_path.read_text().splitlines(keepends=True)
    target=next(x.removeprefix("*** Update File: ").strip() for x in lines if x.startswith("*** Update File: "))
    original=(ROOT/target).read_text()
    chunks=[]
    current=None
    for line in lines:
        if line.startswith("@@"):
            current=[]
            chunks.append(current)
        elif line.startswith("***"):
            continue
        elif current is not None:
            current.append(line)
    transformed=original
    offset=0
    for chunk in chunks:
        old="".join(x[1:] for x in chunk if x[0] in (" ","-"))
        new="".join(x[1:] for x in chunk if x[0] in (" ","+"))
        location=transformed.find(old,offset)
        if location<0 or transformed.find(old,location+1)>=0:
            raise ValueError("missing or ambiguous hunk in "+target)
        transformed=transformed[:location]+new+transformed[location+len(old):]
        offset=location+len(new)
    ast.parse(transformed)
    return dict(target=target,hunks=len(chunks),candidate_sha256=hashlib.sha256(transformed.encode()).hexdigest())


if __name__=="__main__":
    for name in ("actor448_v2.apply_patch.txt","existing_actor_tests448.apply_patch.txt"):
        print(candidate(DIRECTORY/name))
    ast.parse((DIRECTORY/"test_actor448_v2_cold.py").read_text())
    print("Candidates apply uniquely; AST valid. No writes, execution, Torch, or robot imports.")
