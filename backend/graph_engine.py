from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from lxml import etree

XSD_NS = "http://www.w3.org/2001/XMLSchema"
NS_MAP = {"xsd": XSD_NS}


def local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return str(tag)
    if "}" in tag:
        return tag.split("}", 1)[1]
    if ":" in tag:
        return tag.split(":", 1)[1]
    return tag


class XSDSchemaAnalyzer:
    """Parses an XML Schema (.xsd) into a schema dictionary and a Meta-Graph."""

    def __init__(self, xsd_path: Path):
        self.xsd_path = xsd_path
        self.xsd_doc = etree.parse(str(xsd_path))
        self.xml_schema = etree.XMLSchema(self.xsd_doc)
        self.root = self.xsd_doc.getroot()

        self.schema_info: Dict[str, Any] = {
            "filename": xsd_path.name,
            "id": self.root.get("id", ""),
            "version": self.root.get("version", ""),
            "documentation": [],
            "elements": {},
            "groups": {},
        }
        self.schema_graph = nx.DiGraph()
        self._analyze()

    def _analyze(self) -> None:
        for doc in self.root.findall("xsd:annotation/xsd:documentation", NS_MAP):
            if doc.text and doc.text.strip():
                self.schema_info["documentation"].append(doc.text.strip())

        for group_el in self.root.findall("xsd:group[@name]", NS_MAP):
            gname = group_el.get("name")
            if not gname:
                continue
            children = self._extract_particles(group_el)
            self.schema_info["groups"][gname] = {
                "name": gname,
                "children": children,
            }
            group_node_id = f"xsd_group:{gname}"
            self.schema_graph.add_node(
                group_node_id,
                id=group_node_id,
                label=f"Group: {gname}",
                tag="xsd:group",
                node_type="xsd_group",
                properties={"name": gname, "particle_count": len(children)},
            )

        for el_decl in self.root.findall(".//xsd:element[@name]", NS_MAP):
            ename = el_decl.get("name")
            if not ename or ename == "letter":
                continue
            etype = el_decl.get("type", "")
            complex_type = el_decl.find("xsd:complexType", NS_MAP)
            is_mixed = False
            attributes = []
            particles = []

            if complex_type is not None:
                is_mixed = complex_type.get("mixed", "false").lower() == "true"
                particles = self._extract_particles(complex_type)
                flattened_particles = []
                for p in particles:
                    if p["kind"] in ("element_ref", "local_element") and p["target"] == "letter":
                        flattened_particles.append(
                            {
                                "kind": "element_ref",
                                "target": "mainTerm",
                                "minOccurs": p.get("minOccurs", "1"),
                                "maxOccurs": p.get("maxOccurs", "unbounded"),
                                "compositor": p.get("compositor", "sequence"),
                            }
                        )
                    else:
                        flattened_particles.append(p)
                particles = flattened_particles

                for attr_el in complex_type.findall(".//xsd:attribute", NS_MAP):
                    attr_info = self._parse_attribute(attr_el)
                    attributes.append(attr_info)

            has_child_elements = any(
                p["kind"] in ("element_ref", "group_ref", "local_element")
                for p in particles
            )
            if has_child_elements and not is_mixed:
                category = "entity"
            elif is_mixed and has_child_elements:
                category = "mixed_property"
            else:
                category = "scalar_property"

            self.schema_info["elements"][ename] = {
                "name": ename,
                "type": etype or ("complexType" if complex_type is not None else "any"),
                "category": category,
                "is_complex": complex_type is not None,
                "is_mixed": is_mixed,
                "attributes": attributes,
                "particles": particles,
            }

            node_id = f"xsd_el:{ename}"
            self.schema_graph.add_node(
                node_id,
                id=node_id,
                label=ename,
                tag="xsd:element",
                node_type=f"xsd_{category}",
                properties={
                    "name": ename,
                    "type": etype,
                    "category": category,
                    "is_complex": complex_type is not None,
                    "attributes": attributes,
                },
            )

    def _extract_particles(self, parent_el: etree._Element) -> List[Dict[str, Any]]:
        particles: List[Dict[str, Any]] = []
        for child in parent_el.iter():
            if child is parent_el:
                continue
            lname = local_name(child.tag)
            current_compositor = "sequence"
            p_curr = child.getparent()
            while p_curr is not None and p_curr is not parent_el:
                p_lname = local_name(p_curr.tag)
                if p_lname in ("sequence", "choice", "all"):
                    current_compositor = p_lname
                    break
                p_curr = p_curr.getparent()

            if lname == "element":
                ref = child.get("ref")
                name = child.get("name")
                particles.append(
                    {
                        "kind": "element_ref" if ref else "local_element",
                        "target": ref or name or "unknown",
                        "minOccurs": child.get("minOccurs", "1"),
                        "maxOccurs": child.get("maxOccurs", "1"),
                        "compositor": current_compositor,
                    }
                )
            elif lname == "group":
                ref = child.get("ref")
                if ref:
                    particles.append(
                        {
                            "kind": "group_ref",
                            "target": ref,
                            "minOccurs": child.get("minOccurs", "1"),
                            "maxOccurs": child.get("maxOccurs", "1"),
                            "compositor": current_compositor,
                        }
                    )
        return particles

    def _parse_attribute(self, attr_el: etree._Element) -> Dict[str, Any]:
        name = attr_el.get("name", "")
        use = attr_el.get("use", "optional")
        base_type = attr_el.get("type", "xsd:string")
        restrictions: Dict[str, Any] = {}
        restr_el = attr_el.find(".//xsd:restriction", NS_MAP)
        if restr_el is not None:
            base_type = restr_el.get("base", base_type)
            enums = [
                e.get("value")
                for e in restr_el.findall("xsd:enumeration", NS_MAP)
                if e.get("value") is not None
            ]
            if enums:
                restrictions["enumeration"] = enums
            min_inc = restr_el.find("xsd:minInclusive", NS_MAP)
            max_inc = restr_el.find("xsd:maxInclusive", NS_MAP)
            if min_inc is not None:
                restrictions["minInclusive"] = min_inc.get("value")
            if max_inc is not None:
                restrictions["maxInclusive"] = max_inc.get("value")
        return {
            "name": name,
            "use": use,
            "base_type": base_type,
            "restrictions": restrictions,
        }

    def to_cytoscape(self) -> Dict[str, Any]:
        nodes = []
        edges = []
        for nid, data in self.schema_graph.nodes(data=True):
            nodes.append({"data": dict(data)})
        for u, v, data in self.schema_graph.edges(data=True):
            edges.append({"data": {"source": u, "target": v, **data}})
        return {
            "mode": "schema",
            "nodes": nodes,
            "edges": edges,
            "schema_info": self.schema_info,
        }


class XMLGraphWorkspace:
    """Loads an XML Document + XSD Schema, validates it, and indexes it into an interactive graph."""

    REFERENCE_TAGS = {"code", "see", "seeAlso", "manif", "seecat", "subcat"}
    INLINE_PROPERTY_TAGS = {"title", "version", "nemod", "warning"}

    def __init__(self, xml_path: Path, xsd_path: Path):
        self.xml_path = xml_path
        self.xsd_path = xsd_path
        t0 = time.perf_counter()

        self.schema_analyzer = XSDSchemaAnalyzer(xsd_path)
        self.xml_doc = etree.parse(str(xml_path))
        self.is_valid = self.schema_analyzer.xml_schema.validate(self.xml_doc)
        self.validation_errors = [
            {
                "line": err.line,
                "column": err.column,
                "message": err.message,
                "level": err.level_name,
            }
            for err in list(self.schema_analyzer.xml_schema.error_log)[:50]
        ]

        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.parent_to_children: Dict[str, List[str]] = defaultdict(list)
        self.semantic_edges: List[Dict[str, Any]] = []
        self.node_semantic_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        self.root_id: str = "n_0"
        self.main_term_by_title: Dict[str, str] = {}
        self.code_to_node_id: Dict[str, str] = {}
        self.code_to_terms: Dict[str, List[str]] = defaultdict(list)
        self.tag_counts: Counter = Counter()

        self.trajectories: List[Dict[str, Any]] = []
        self.node_to_trajectory_ids: Dict[str, List[int]] = defaultdict(list)
        self.node_search_text: Dict[str, str] = {}
        self.node_title_lower: Dict[str, str] = {}

        self._build_index()
        self.load_time_ms = round((time.perf_counter() - t0) * 1000, 1)

    def _is_entity_element(self, el: etree._Element) -> bool:
        tag = local_name(el.tag)
        el_schema = self.schema_analyzer.schema_info["elements"].get(tag)
        if el_schema:
            return el_schema["category"] == "entity"
        child_tags = [local_name(c.tag) for c in el if isinstance(c.tag, str)]
        return len(child_tags) > 0 and tag not in self.INLINE_PROPERTY_TAGS

    def _extract_title_and_nemod(self, el: etree._Element) -> Tuple[str, str]:
        title_el = el.find("title")
        if title_el is not None:
            base_text = (title_el.text or "").strip()
            nemods = []
            tail_parts = []
            for child in title_el:
                if local_name(child.tag) == "nemod" and child.text:
                    nemods.append(child.text.strip())
                if child.tail and child.tail.strip():
                    tail_parts.append(child.tail.strip())
            full_title = " ".join([p for p in [base_text] + tail_parts if p]).strip()
            nemod_str = " ".join(nemods).strip()
            return full_title or nemod_str, nemod_str

        for attr_key in ("name", "id", "title", "label", "code", "col"):
            if el.get(attr_key):
                return f"{local_name(el.tag)} [{el.get(attr_key)}]", ""
        if el.text and el.text.strip():
            txt = el.text.strip()
            return (txt[:45] + "…") if len(txt) > 45 else txt, ""
        return local_name(el.tag), ""

    def _build_index(self) -> None:
        root_el = self.xml_doc.getroot()
        counter = 0

        for el in self.xml_doc.iter():
            if isinstance(el.tag, str):
                tname = local_name(el.tag)
                if tname != "letter":
                    self.tag_counts[tname] += 1

        def walk(el: etree._Element, parent_id: Optional[str], depth: int, path_titles: List[str]) -> str:
            nonlocal counter
            node_id = f"n_{counter}"
            counter += 1

            tag = local_name(el.tag)
            title, nemod = self._extract_title_and_nemod(el)
            attrs = {k: v for k, v in el.attrib.items()}
            props: Dict[str, Any] = {}
            ref_items: List[Tuple[str, str]] = []

            entity_children: List[etree._Element] = []
            for child in el:
                if not isinstance(child.tag, str):
                    continue
                ctag = local_name(child.tag)
                if ctag == "letter":
                    for grand_child in child:
                        if isinstance(grand_child.tag, str) and self._is_entity_element(grand_child):
                            entity_children.append(grand_child)
                elif self._is_entity_element(child):
                    entity_children.append(child)
                elif ctag == "title":
                    props["title"] = title
                    if nemod:
                        props["nemod"] = nemod
                else:
                    val = "".join(child.itertext()).strip()
                    if val:
                        if ctag in props:
                            if isinstance(props[ctag], list):
                                props[ctag].append(val)
                            else:
                                props[ctag] = [props[ctag], val]
                        else:
                            props[ctag] = val
                        if ctag in self.REFERENCE_TAGS:
                            ref_items.append((ctag, val))

            if tag == "ICD10CM.index":
                display_label = f"{title or 'ICD-10-CM Index'} (v{props.get('version', '')})"
            elif tag == "mainTerm":
                display_label = title
                if title:
                    self.main_term_by_title[title.lower()] = node_id
            elif tag == "term":
                lvl = attrs.get("level", "")
                display_label = f"{title}" if title else f"term (L{lvl})"
            else:
                display_label = title or tag

            current_path = path_titles + ([title] if title else [tag])

            node_data: Dict[str, Any] = {
                "id": node_id,
                "tag": tag,
                "node_type": tag if tag in ("ICD10CM.index", "mainTerm", "term") else "entity",
                "label": display_label,
                "title": title,
                "nemod": nemod,
                "level": int(attrs["level"]) if "level" in attrs and attrs["level"].isdigit() else depth,
                "depth": depth,
                "parent_id": parent_id,
                "attributes": attrs,
                "properties": props,
                "line": el.sourceline or 1,
                "breadcrumb": " › ".join(current_path),
                "child_count": len(entity_children),
                "has_children": len(entity_children) > 0 or len(ref_items) > 0,
            }
            self.nodes[node_id] = node_data
            if parent_id is not None:
                self.parent_to_children[parent_id].append(node_id)

            for ref_tag, ref_val in ref_items:
                if ref_tag in ("code", "manif", "seecat", "subcat"):
                    code_node_id = f"{ref_tag}:{ref_val}"
                    if code_node_id not in self.nodes:
                        code_label = f"Code: {ref_val}" if ref_tag == "code" else f"{ref_tag.upper()}: {ref_val}"
                        self.nodes[code_node_id] = {
                            "id": code_node_id,
                            "tag": ref_tag,
                            "node_type": ref_tag,
                            "label": code_label,
                            "title": ref_val,
                            "nemod": "",
                            "level": depth + 1,
                            "depth": depth + 1,
                            "parent_id": node_id,
                            "attributes": {},
                            "properties": {ref_tag: ref_val},
                            "line": el.sourceline or 1,
                            "breadcrumb": f"{node_data['breadcrumb']} → {code_label}",
                            "child_count": 0,
                            "has_children": False,
                        }
                    if ref_tag == "code":
                        self.code_to_node_id[ref_val] = code_node_id
                        self.code_to_terms[code_node_id].append(node_id)

                    edge = {
                        "id": f"e_{node_id}_{code_node_id}",
                        "source": node_id,
                        "target": code_node_id,
                        "label": "CODES_TO" if ref_tag == "code" else ref_tag.upper(),
                        "edge_type": "CODES_TO" if ref_tag == "code" else ref_tag.upper(),
                    }
                    self.node_semantic_edges[node_id].append(edge)
                    self.node_semantic_edges[code_node_id].append(edge)
                elif ref_tag in ("see", "seeAlso"):
                    self.semantic_edges.append(
                        {
                            "source": node_id,
                            "ref_tag": ref_tag,
                            "ref_text": ref_val,
                            "line": el.sourceline or 1,
                        }
                    )

            for child_el in entity_children:
                walk(child_el, node_id, depth + 1, current_path)

            return node_id

        self.root_id = walk(root_el, None, 0, [])

        for item in self.semantic_edges:
            src_id = item["source"]
            ref_tag = item["ref_tag"]
            ref_text = item["ref_text"]
            edge_type = "SEE" if ref_tag == "see" else "SEE_ALSO"

            head_candidate = ref_text.split(",")[0].strip().lower()
            target_id = self.main_term_by_title.get(ref_text.lower()) or self.main_term_by_title.get(head_candidate)

            if not target_id:
                target_id = f"ref:{ref_tag}:{ref_text}"
                if target_id not in self.nodes:
                    self.nodes[target_id] = {
                        "id": target_id,
                        "tag": ref_tag,
                        "node_type": ref_tag,
                        "label": f"{'See' if ref_tag == 'see' else 'See Also'}: {ref_text}",
                        "title": ref_text,
                        "nemod": "",
                        "level": 2,
                        "depth": 2,
                        "parent_id": None,
                        "attributes": {},
                        "properties": {ref_tag: ref_text},
                        "line": item["line"],
                        "breadcrumb": f"Reference › {ref_text}",
                        "child_count": 0,
                        "has_children": False,
                    }

            edge = {
                "id": f"e_{src_id}_{target_id}_{edge_type}",
                "source": src_id,
                "target": target_id,
                "label": edge_type,
                "edge_type": edge_type,
                "ref_text": ref_text,
            }
            self.node_semantic_edges[src_id].append(edge)
            self.node_semantic_edges[target_id].append(edge)

        for code_node_id, term_ids in self.code_to_terms.items():
            if code_node_id in self.nodes:
                self.nodes[code_node_id]["child_count"] = len(term_ids)
                self.nodes[code_node_id]["properties"]["linked_terms_count"] = len(term_ids)

        for nid, ndata in self.nodes.items():
            if nid == self.root_id:
                continue
            title_val = (ndata.get("title") or "").strip().lower()
            self.node_title_lower[nid] = title_val
            self.node_search_text[nid] = title_val

        for nid, ndata in self.nodes.items():
            if ndata["tag"] not in ("mainTerm", "term", "entity"):
                continue
            for sedge in self.node_semantic_edges.get(nid, []):
                if sedge["source"] == nid and sedge["edge_type"] in ("CODES_TO", "MANIF", "SEECAT", "SUBCAT"):
                    code_node_id = sedge["target"]
                    chain: List[str] = []
                    curr: Optional[str] = nid
                    while curr is not None:
                        chain.append(curr)
                        curr = self.nodes.get(curr, {}).get("parent_id")
                    chain.reverse()
                    traj_nodes = tuple(chain + [code_node_id])
                    hier_edges = [f"e_hier_{chain[i]}_{chain[i + 1]}" for i in range(len(chain) - 1)]
                    traj_edges = tuple(hier_edges + [sedge["id"]])
                    t_idx = len(self.trajectories)
                    code_node = self.nodes.get(code_node_id, {})
                    code_label = code_node.get("label", code_node_id)
                    self.trajectories.append(
                        {
                            "id": t_idx,
                            "nodes": traj_nodes,
                            "edges": traj_edges,
                            "term_id": nid,
                            "code_id": code_node_id,
                            "code_edge_id": sedge["id"],
                            "code": code_node.get("title", ""),
                            "breadcrumb": f"{ndata.get('breadcrumb', '')} → {code_label}",
                        }
                    )
                    for path_nid in traj_nodes[1:]:
                        self.node_to_trajectory_ids[path_nid].append(t_idx)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "xml_filename": self.xml_path.name,
            "xml_size_bytes": self.xml_path.stat().st_size,
            "xsd_filename": self.xsd_path.name,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "load_time_ms": self.load_time_ms,
            "total_graph_nodes": len(self.nodes),
            "total_trajectories": len(self.trajectories),
            "tag_counts": dict(self.tag_counts),
            "schema_info": self.schema_analyzer.schema_info,
        }

    def _add_ancestor_hierarchy(self, node_ids: Set[str]) -> Set[str]:
        with_ancestors = set(node_ids)
        for nid in list(node_ids):
            curr = self.nodes.get(nid, {}).get("parent_id")
            while curr:
                with_ancestors.add(curr)
                curr = self.nodes.get(curr, {}).get("parent_id")
        return with_ancestors

    def _build_subgraph_payload(
        self,
        included_node_ids: Set[str],
        include_semantic: bool = True,
        highlight_ids: Optional[Set[str]] = None,
        allowed_semantic_edge_ids: Optional[Set[str]] = None,
        node_match_details: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        highlight_ids = highlight_ids or set()
        node_match_details = node_match_details or {}
        cy_nodes = []
        cy_edges = []
        seen_edges: Set[str] = set()

        for nid in included_node_ids:
            node = self.nodes.get(nid)
            if not node:
                continue
            is_hl = nid in highlight_ids
            ndata = {
                "id": node["id"],
                "label": node["label"],
                "tag": node["tag"],
                "node_type": node["node_type"],
                "level": node.get("level", 0),
                "depth": node.get("depth", 0),
                "title": node.get("title", ""),
                "nemod": node.get("nemod", ""),
                "breadcrumb": node.get("breadcrumb", ""),
                "child_count": node.get("child_count", 0),
                "has_children": node.get("has_children", False),
                "line": node.get("line", 1),
                "is_highlight": is_hl,
                "match_occurrences": node_match_details.get(nid, {}).get("occurrences", 0),
                "matched_keywords": node_match_details.get(nid, {}).get("matched_keywords", []),
            }
            cy_nodes.append({"data": ndata})

            pid = node.get("parent_id")
            if pid and pid in included_node_ids:
                eid = f"e_hier_{pid}_{nid}"
                if eid not in seen_edges:
                    seen_edges.add(eid)
                    edge_label = f"HAS_{node['tag'].upper()}"
                    if node["tag"] == "term":
                        edge_label = f"SUBTERM (L{node.get('level', 1)})"
                    cy_edges.append(
                        {
                            "data": {
                                "id": eid,
                                "source": pid,
                                "target": nid,
                                "label": edge_label,
                                "edge_type": "HIERARCHY",
                            }
                        }
                    )

            for sedge in self.node_semantic_edges.get(nid, []):
                if sedge["source"] in included_node_ids and sedge["target"] in included_node_ids:
                    if allowed_semantic_edge_ids is not None and sedge["id"] not in allowed_semantic_edge_ids:
                        continue
                    if sedge["id"] not in seen_edges:
                        seen_edges.add(sedge["id"])
                        cy_edges.append({"data": dict(sedge)})

        return {
            "mode": "document",
            "nodes": cy_nodes,
            "edges": cy_edges,
            "node_count": len(cy_nodes),
            "edge_count": len(cy_edges),
        }

    def search_subgraph(
        self, query: str, max_results: int = 15, include_children: bool = True
    ) -> Dict[str, Any]:
        raw_keywords = [k.strip() for k in query.split(";") if k.strip()]
        if not raw_keywords:
            return {"mode": "document", "nodes": [], "edges": [], "node_count": 0, "edge_count": 0, "trajectories": []}

        keywords: List[Tuple[str, str, Optional[re.Pattern]]] = []
        seen_kw: Set[str] = set()
        for rk in raw_keywords:
            kl = rk.lower()
            if kl not in seen_kw:
                seen_kw.add(kl)
                prefix = r"\b" if kl[0].isalnum() else ""
                suffix = r"\b" if kl[-1].isalnum() else ""
                pat = re.compile(rf"{prefix}{re.escape(kl)}{suffix}") if (prefix or suffix) else None
                keywords.append((kl, rk, pat))

        use_word_boundary: List[bool] = []
        for kl, _, pat in keywords:
            if pat is None:
                use_word_boundary.append(False)
            else:
                has_wb = any(pat.search(stext) for stext in self.node_search_text.values() if kl in stext)
                use_word_boundary.append(has_wb)

        node_match_details: Dict[str, Dict[str, Any]] = {}
        traj_occurrences: Counter = Counter()
        traj_kw_mask: Dict[int, int] = defaultdict(int)
        traj_exact_hits: Counter = Counter()

        for nid, stext in self.node_search_text.items():
            if not stext:
                continue
            t_ids = self.node_to_trajectory_ids.get(nid)
            if not t_ids:
                continue
            occ_count = 0
            kw_mask = 0
            exact_count = 0
            matched_kws: List[str] = []
            for kw_idx, (kl, kdisp, pat) in enumerate(keywords):
                if kl not in stext:
                    continue
                if use_word_boundary[kw_idx] and pat is not None:
                    matched = bool(pat.search(stext))
                else:
                    matched = True
                if matched:
                    occ_count += 1
                    kw_mask |= 1 << kw_idx
                    matched_kws.append(kdisp)
                    if stext == kl:
                        exact_count += 1
            if occ_count > 0:
                node_match_details[nid] = {
                    "occurrences": occ_count,
                    "matched_keywords": matched_kws,
                }
                for tid in t_ids:
                    traj_occurrences[tid] += occ_count
                    traj_kw_mask[tid] |= kw_mask
                    if exact_count > 0:
                        traj_exact_hits[tid] += exact_count

        all_matching_tids = set(traj_kw_mask.keys())
        if not all_matching_tids:
            return {
                "mode": "document",
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "matched_count": 0,
                "total_matching_trajectories": 0,
                "top_n": max_results,
                "keywords": [kdisp for _, kdisp, _ in keywords],
                "trajectories": [],
                "query": query,
            }

        ranked_tids = sorted(
            all_matching_tids,
            key=lambda tid: (
                traj_occurrences[tid],
                traj_kw_mask[tid].bit_count(),
                traj_exact_hits.get(tid, 0),
                -len(self.trajectories[tid]["nodes"]),
                -tid,
            ),
            reverse=True,
        )

        top_tids = ranked_tids[:max_results]
        included_nodes: Set[str] = set()
        allowed_semantic_edges: Set[str] = set()
        highlighted_nodes: Set[str] = set()
        ranked_trajectories_payload: List[Dict[str, Any]] = []

        for rank_idx, tid in enumerate(top_tids, start=1):
            traj = self.trajectories[tid]
            for nid in traj["nodes"]:
                included_nodes.add(nid)
                if nid in node_match_details:
                    highlighted_nodes.add(nid)
            allowed_semantic_edges.add(traj["code_edge_id"])

            kw_mask = traj_kw_mask[tid]
            traj_kws = [kdisp for idx, (_, kdisp, _) in enumerate(keywords) if (kw_mask & (1 << idx))]

            ranked_trajectories_payload.append(
                {
                    "rank": rank_idx,
                    "trajectory_id": tid,
                    "keyword_occurrences": traj_occurrences[tid],
                    "distinct_keywords_matched": kw_mask.bit_count(),
                    "matched_keywords": traj_kws,
                    "code": traj["code"],
                    "code_id": traj["code_id"],
                    "term_id": traj["term_id"],
                    "breadcrumb": traj["breadcrumb"],
                    "nodes": list(traj["nodes"]),
                    "edges": list(traj["edges"]),
                }
            )

        res = self._build_subgraph_payload(
            included_nodes,
            include_semantic=False,
            highlight_ids=highlighted_nodes,
            allowed_semantic_edge_ids=allowed_semantic_edges,
            node_match_details=node_match_details,
        )
        res["matched_count"] = len(ranked_trajectories_payload)
        res["total_matching_trajectories"] = len(ranked_tids)
        res["top_n"] = max_results
        res["keywords"] = [kdisp for _, kdisp, _ in keywords]
        res["trajectories"] = ranked_trajectories_payload
        res["matched_ids"] = list(highlighted_nodes)
        res["query"] = query
        return res

    def get_fork_subgraph(
        self,
        trajectory_ids: List[int],
        highlight_fork_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Builds a focused Cytoscape.js subgraph representing competing trajectories
        and their branch points (forks) up to the common ancestor.
        """
        if not trajectory_ids:
            return {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0}

        included_nodes: Set[str] = set()
        allowed_semantic_edges: Set[str] = set()
        highlighted_nodes: Set[str] = set()

        if highlight_fork_node_id:
            highlighted_nodes.add(highlight_fork_node_id)

        for tid in trajectory_ids:
            if tid < 0 or tid >= len(self.trajectories):
                continue
            traj = self.trajectories[tid]
            for nid in traj["nodes"]:
                included_nodes.add(nid)
            allowed_semantic_edges.add(traj["code_edge_id"])
            highlighted_nodes.add(traj["code_id"])

        res = self._build_subgraph_payload(
            included_nodes,
            include_semantic=False,
            highlight_ids=highlighted_nodes,
            allowed_semantic_edge_ids=allowed_semantic_edges,
        )
        res["trajectory_ids"] = trajectory_ids
        res["fork_node_id"] = highlight_fork_node_id
        return res

    def get_node_detail(self, node_id: str) -> Dict[str, Any]:
        node = self.nodes.get(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}

        children_preview = [
            {
                "id": cid,
                "label": self.nodes[cid]["label"],
                "tag": self.nodes[cid]["tag"],
                "code": self.nodes[cid].get("properties", {}).get("code", ""),
            }
            for cid in self.parent_to_children.get(node_id, [])[:100]
            if cid in self.nodes
        ]
        return {
            "node": node,
            "children_preview": children_preview,
            "total_children": len(self.parent_to_children.get(node_id, [])),
            "semantic_links": self.node_semantic_edges.get(node_id, [])[:30],
        }
