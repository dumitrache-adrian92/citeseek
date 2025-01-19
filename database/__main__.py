from .paper_database import PaperDatabase

DATABASE_URL = "http://localhost:9200"


def populate_database():
    db = PaperDatabase(DATABASE_URL)
    db.reindex("paper_abstracts")


if __name__ == '__main__':
    populate_database()
