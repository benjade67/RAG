# Architecture cible

## Objectif

Construire un RAG specialise sur des plans AutoCAD en PDF, capable de repondre avec citations et d'exploiter le texte ainsi que la structure visuelle.

## Vue logique

1. Ingestion
2. Extraction texte + layout
3. Structuration en regions visuelles
4. Chunking layout-aware
5. Indexation hybride
6. Retrieval + reranking
7. Generation contrainte avec citations

## Modules

### Domaine

- `models.py`: objets metier purs
- `ports.py`: contrats abstraits

### Application

- orchestration des cas d'usage
- aucune dependance a une techno concrete

### Infrastructure

- parseurs PDF
- index vectoriel
- moteur lexical
- reranker
- LLM / generation

## Pourquoi cette decomposition respecte SOLID

### Single Responsibility

Chaque classe porte un role unique:

- parser un document
- segmenter
- vectoriser
- indexer
- retrouver
- generer

### Open/Closed

Le systeme est ouvert a l'ajout de nouvelles implementations sans modifier les cas d'usage:

- autre parseur PDF
- autre moteur d'embeddings
- autre vector store
- autre LLM

### Liskov Substitution

Les services applicatifs manipulent des abstractions (`DocumentParser`, `ChunkIndex`, `AnswerGenerator`) et peuvent remplacer une implementation par une autre.

### Interface Segregation

Les contrats sont petits et focalises.

### Dependency Inversion

Le coeur applicatif depend des ports, jamais des composants techniques concrets.

## Strategie layout-aware recommandee pour les plans

Le chunking doit etre structurel, pas seulement textuel.

Un chunk ideal contient:

- texte local
- region visuelle
- page
- bbox
- type de region
- voisinage eventuel

Exemples de regions:

- cartouche
- tableau de nomenclature
- note de revision
- annotation
- repere de zone
- legende

## Strategie de citations

Chaque chunk doit pouvoir etre restitue avec:

- `document_id`
- `page_number`
- `bbox`
- `excerpt`

L'UI pourra ensuite:

- surligner la zone du PDF
- ouvrir la bonne page
- afficher les preuves utilisees

## Recommandations techniques pour la suite

### Parsing PDF

Priorite a un parseur capable de sortir:

- texte
- mots ou lignes
- coordonnees
- ordre de lecture
- tableaux / blocs si possible

Le projet contient une premiere implementation `PyMuPdfParser` qui:

- lit les blocs textuels via `page.get_text("dict")`
- extrait des lignes ordonnees
- annote chaque region avec une heuristique de type (`title_block`, `table`, `note`, `callout`, `free_text`)
- conserve la `bbox` de chaque bloc pour les citations

### Indexation

Index hybride conseille:

- vectoriel pour la semantique
- lexical pour references exactes, tags, identifiants, repereurs

### Generation

Le generateur doit imposer:

- interdiction d'inventer
- reponse uniquement a partir des passages recuperes
- citations obligatoires
- mention explicite d'absence de preuve si necessaire

## Roadmap

1. Connecter un parseur PDF numerique reel
2. Ajouter un corpus de plans de test
3. Mesurer precision retrieval/citation
4. Ajouter extraction ciblee de valeurs
5. Ajouter support OCR/scans
