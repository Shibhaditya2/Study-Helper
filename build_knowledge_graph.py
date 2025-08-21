import csv
import time
import os
import logging
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, SKOS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

NS = Namespace("http://www.wikipedia.org")
SKOS_NS = SKOS

def build_kg(csv_file="data/concepts.csv"):
    start_time = time.time()
    logger.info(f"Building knowledge graph from {csv_file}")
    
    g = Graph()
    g.bind("skos", SKOS_NS)
    g.bind("ns", NS)
    
    concept_count = 0
    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                concept_count += 1
                concept = URIRef(NS[row["concept"].replace(" ", "_")])
                prereq = URIRef(NS[row["prerequisite"].replace(" ", "_")]) if row["prerequisite"] else None
                
                g.add((concept, RDF.type, SKOS_NS.Concept))
                g.add((concept, SKOS.prefLabel, Literal(row["concept"])))
                
                if prereq:
                    g.add((prereq, RDF.type, SKOS_NS.Concept))
                    g.add((prereq, SKOS.prefLabel, Literal(row["prerequisite"])))
                    g.add((concept, SKOS.related, prereq))
    
        g.serialize("knowledge_graph.ttl", format="turtle")
        elapsed_time = time.time() - start_time
        triple_count = len(g)
        file_size = os.path.getsize("knowledge_graph.ttl") / 1024 / 1024  # MB
        logger.info(f"Knowledge graph built: {concept_count} concepts, {triple_count} triples, time={elapsed_time:.2f}s, file_size={file_size:.2f}MB")
        return g
    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}")
        return None