from rdflib import Graph
from rdflib.namespace import SKOS
from rdflib import URIRef

def load_kg(file="knowledge_graph.ttl"):
    g = Graph()
    try:
        g.parse(file, format="turtle")
        return g
    except Exception as e:
        print(f"Error loading KG: {e}")
        return None

def traverse_kg(graph, query_concept):
    if not graph:
        return []
    concept_uri = URIRef(f"http://example.org/studyhelper/{query_concept.replace(' ', '_')}")
    prerequisites = []
    for s, p, o in graph.triples((concept_uri, SKOS.related, None)):
        prereq_label = graph.value(o, SKOS.prefLabel)
        if prereq_label:
            prerequisites.append(str(prereq_label))
    return prerequisites