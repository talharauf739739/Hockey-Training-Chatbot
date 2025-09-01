import sqlite3
import logging
import pickle
from sentence_transformers import SentenceTransformer
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Database and model setup
DATABASE_PATH = os.getenv("DATABASE_PATH", "HockeyFood.db")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Core hockey keywords for text augmentation
hockey_keywords = [
    "hockey", "field hockey", "veldhockey", "match", "wedstrijd", "game", "spel", "goal", "doelpunt",
    "score", "scoren", "ball", "bal", "stick", "hockeystick", "field", "veld", "turf", "kunstgras",
    "pitch", "speelveld", "corner", "short corner", "long corner", "korte hoek", "lange hoek",
    "penalty", "strafbal", "shootout", "strookschot", "penalty stroke", "strafslag",
    "coach", "trainer", "goalkeeper", "doelman", "keeper", "goalie", "defender", "verdediger",
    "midfielder", "middenvelder", "forward", "aanvaller", "striker", "spits", "captain", "aanvoerder",
    "player", "speler", "team", "ploeg",
    "shooting", "schieten", "passing", "passen", "backhand", "achterhand", "forehand", "voorhand",
    "wrist shot", "pols schot", "slap shot", "slagschot", "drag flick", "sleeppush", "push pass",
    "pushpass", "hit pass", "slagpass", "aerial pass", "luchtpass", "dribbling", "dribbelen",
    "stick work", "stickwerk", "deflection", "afbuiging", "scoop", "scheppen", "tackle", "tackelen",
    "block tackle", "blok tackle", "jab tackle", "steektackle", "reverse stick", "omgekeerde stick",
    "indian dribble", "indiase dribbel", "3d skills", "3d vaardigheden", "goalkeeping", "doelverdediging",
    "save", "redding", "clearance", "uitverdediging", "flick", "slepen", "lift", "optillen",
    "chip", "chippen", "sweep hit", "veegslag", "tomahawk", "backstick", "reverse hit", "omgekeerde slag",
    "drag", "slepen", "dummy", "schijnbeweging", "feint", "fint", "spin", "draaien",
    "training", "oefening", "exercise", "oefenen", "drill", "oefensessie", "practice", "praktijk",
    "warm-up", "opwarming", "cool-down", "afkoeling", "conditioning", "conditietraining",
    "fitness", "fitheid", "agility", "wendbaarheid", "speed", "snelheid", "endurance", "uithoudingsvermogen",
    "strength", "kracht", "core strength", "kernkracht", "stick handling", "stickbeheersing",
    "ball control", "balbeheersing", "footwork", "voetwerk", "positioning", "positionering",
    "marking", "dekken", "zone defense", "zonedekking", "man-to-man", "man-op-man",
    "attack drill", "aanvalsoefening", "defense drill", "verdedigingsoefening"
]

def initialize_database():
    """Update YouTube_Urls table with id as PRIMARY KEY, embeddings, and create FTS5 table."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Check if id column exists
        cursor.execute("PRAGMA table_info(YouTube_Urls)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'id' not in columns:
            # Create new table with id as PRIMARY KEY
            cursor.execute("""
                CREATE TABLE YouTube_Urls_New (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    url TEXT,
                    metatags TEXT,
                    embedding BLOB
                )
            """)
            # Copy data from old table
            cursor.execute("INSERT INTO YouTube_Urls_New (title, url, metatags) SELECT title, url, metatags FROM YouTube_Urls")
            # Drop old table and rename new one
            cursor.execute("DROP TABLE YouTube_Urls")
            cursor.execute("ALTER TABLE YouTube_Urls_New RENAME TO YouTube_Urls")
            conn.commit()
        elif 'embedding' not in columns:
            # Add embedding column if id exists but embedding doesn't
            cursor.execute("ALTER TABLE YouTube_Urls ADD COLUMN embedding BLOB")
            conn.commit()

        # Create FTS5 table
        cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS YouTube_Urls_FTS USING fts5(title, metatags)")

        # Fetch all rows and compute embeddings
        cursor.execute("SELECT id, title, metatags FROM YouTube_Urls")
        rows = cursor.fetchall()
        fts_data = []
        for row in rows:
            id, title, metatags = row
            if id is None:
                logging.warning(f"Skipping row with null id: {title[:50]}...")
                continue
            text = f"{title} {metatags or ''} {' '.join(hockey_keywords)}".strip()
            if not text:
                continue
            # Compute and store embedding
            embedding = sentence_model.encode(text, normalize_embeddings=True)
            cursor.execute("UPDATE YouTube_Urls SET embedding = ? WHERE id = ?", 
                          (pickle.dumps(embedding), id))
            fts_data.append((id, title, metatags or ''))
            logging.debug(f"Processed row {id}: {title[:50]}...")

        # Populate FTS table with all rows
        if fts_data:
            cursor.executemany("INSERT OR REPLACE INTO YouTube_Urls_FTS (rowid, title, metatags) VALUES (?, ?, ?)", fts_data)

        conn.commit()
        logging.info(f"Processed {len(rows)} rows, stored {len(fts_data)} embeddings and FTS entries.")
        conn.close()

    except sqlite3.Error as e:
        logging.error(f"Database error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    if not os.path.exists(DATABASE_PATH):
        logging.error(f"Database file not found at {DATABASE_PATH}.")
        raise FileNotFoundError(f"Database file not found at {DATABASE_PATH}.")
    initialize_database()