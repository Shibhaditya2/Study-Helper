import csv
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, SKOS

NS = Namespace("http://example.org/studyhelper/")
SKOS_NS = SKOS

def build_kg(csv_file="data/concepts.csv"):
    g = Graph()
    g.bind("skos", SKOS_NS)
    g.bind("ns", NS)
    
    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                concept = URIRef(NS[row["concept"].replace(" ", "_")])
                prereq = URIRef(NS[row["prerequisite"].replace(" ", "_")]) if row["prerequisite"] else None
                
                g.add((concept, RDF.type, SKOS_NS.Concept))
                g.add((concept, SKOS.prefLabel, Literal(row["concept"])))
                
                if prereq:
                    g.add((prereq, RDF.type, SKOS_NS.Concept))
                    g.add((prereq, SKOS.prefLabel, Literal(row["prerequisite"])))
                    g.add((concept, SKOS.related, prereq))
    
        g.serialize("knowledge_graph.ttl", format="turtle")
        print("Knowledge graph built.")
        return g
    except Exception as e:
        print(f"Error building KG: {e}")
        return None