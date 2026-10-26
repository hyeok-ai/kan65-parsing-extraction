"""온톨로지 클래스 후보 및 속성 정의 조회 (설계안 8·10절).

초기에는 임시 JSON(data/demo_ontology.json)을 사용한다.
GraphDB 연동 시 같은 두 메서드를 구현한 소스로 교체하면 02·03 노드는 수정하지 않아도 된다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

DEFAULT_ONTOLOGY = Path(__file__).resolve().parent.parent / "data" / "demo_ontology.json"

# GraphDB 연동 시 사용할 클래스 후보 조회 쿼리 (설계안 8.3). 실제 IRI에 맞춰 변경 필요.
CLASS_CANDIDATES_SPARQL = """
PREFIX ex: <http://example.org/ontology#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?class ?label ?description WHERE {
  VALUES ?class { ex:Motor ex:Bolt ex:Cable ex:Bearing }
  ?class a owl:Class .
  OPTIONAL { ?class rdfs:label ?label . }
  OPTIONAL { ?class rdfs:comment ?description . }
}
"""


class OntologySource(Protocol):
    def class_candidates(self) -> list[dict]: ...

    def properties_of(self, class_id: str) -> list[dict]: ...


class JsonOntology:
    def __init__(self, path: str | Path = DEFAULT_ONTOLOGY):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._classes = {c["id"]: c for c in data["classes"]}

    def class_candidates(self) -> list[dict]:
        return [
            {
                "id": c["id"],
                "label": c["label"],
                "description": c.get("description", ""),
                "synonyms": c.get("synonyms", []),
            }
            for c in self._classes.values()
        ]

    def properties_of(self, class_id: str) -> list[dict]:
        if class_id not in self._classes:
            raise KeyError(f"정의되지 않은 클래스: {class_id}")
        return [dict(p) for p in self._classes[class_id]["properties"]]
