# RAG pour plans AutoCAD PDF

Ce projet pose un socle de conception pour un systeme de Retrieval-Augmented Generation (RAG) cible sur des plans AutoCAD exportes en PDF.

## Cas d'usage cibles

- Recherche documentaire dans un corpus de plans
- Questions/reponses metier avec citations
- Extraction de valeurs cibles (dimensions, reperes, references, pieces, zones)

## Contraintes du domaine

- Les PDF contiennent du texte, mais aussi une structure visuelle importante
- La granularite doit permettre de citer precisement la source
- Certaines questions dependront de la topologie visuelle: cartouche, zones, tableaux, legendes, annotations, reperes
- Le systeme doit pouvoir evoluer d'un mode "texte uniquement" vers un mode "multimodal layout-aware"

## Principes d'architecture

- SOLID
- Separation nette entre domaine, application et infrastructure
- Contrats explicites pour les extracteurs, index, rerankers et generateurs
- Citations de premier niveau sur chaque reponse
- Tracabilite de chaque evidence vers `document_id`, `page_number`, `bbox`, `chunk_id`

## Strategie recommandee

### Phase 1

RAG "layout-aware" sur PDF numeriques:

- Extraction texte + positions par bloc/ligne/mot
- Detection de regions utiles: cartouche, tableaux, annotations, zones techniques
- Chunking structurel par region plutot que par simple fenetre de tokens
- Embeddings hybrides:
  - semantiques pour le texte
  - metadata/layout pour conserver la structure
- Retrieval hybride:
  - vectoriel
  - BM25 / lexical
  - filtres metadata
- Reranking
- Generation avec citations obligatoires

### Phase 2

Ajout d'elements multimodaux:

- OCR si PDF scanne
- Vision/layout model pour mieux exploiter tableaux et reperes graphiques
- Index d'elements visuels (symboles, fleches, labels, proximite spatiale)

## Reponse avec citations

Chaque reponse doit:

- annoncer le niveau de confiance si utile
- citer les passages utilises
- renvoyer les preuves minimales necessaires:
  - document
  - page
  - zone/bbox
  - extrait

Exemple:

```text
La vanne V-204 est associee a la ligne principale de distribution [DOC-12 p.3 zone=cartouche-item-7].
```

## Structure du projet

Voir `src/rag_pdf`.

## Prochaine etape recommandee

1. Brancher un parseur PDF numerique avec layout
2. Ajouter un moteur d'indexation local
3. Implementer la chaine de citations de bout en bout

## Parsing PDF reel

Le projet inclut maintenant un parseur base sur PyMuPDF pour:

- extraire le texte avec coordonnees
- reconstruire des regions par bloc textuel
- poser des heuristiques simples sur la structure visuelle

Installation locale:

```bash
pip install -e .
```

Execution:

```bash
python -m rag_pdf.main chemin/vers/plan.pdf "Quelle est la reference du plan ?"
```

## Interface Streamlit

Une interface web locale est disponible:

```bash
streamlit run streamlit_app.py
```

Elle permet de:

- charger un PDF
- poser une question
- lire la reponse citee
- inspecter les passages recuperes

## Generation Mistral

La generation peut maintenant utiliser l'API Mistral si `MISTRAL_API_KEY` est definie.

Variables d'environnement:

```bash
set MISTRAL_API_KEY=votre_cle
set MISTRAL_MODEL=mistral-small-latest
```

Si `MISTRAL_API_KEY` n'est pas definie, le projet garde le generateur local de fallback.

## Generation locale avec Ollama

Si vous ne disposez pas encore d'une cle API, le projet peut utiliser Ollama en local.

Variables d'environnement:

```bash
set OLLAMA_MODEL=qwen2.5:7b-instruct
set OLLAMA_BASE_URL=http://127.0.0.1:11434
set OLLAMA_TIMEOUT_SECONDS=180
```

Priorite de selection du generateur:

1. Mistral si `MISTRAL_API_KEY` est definie
2. Ollama si `OLLAMA_MODEL` est defini
3. Fallback local sinon

Le projet charge automatiquement le fichier `.env` au demarrage.
