# Plan d'action — RAG local (PDF/TXT/MD) + UI d'import

## 1) Objectif produit
Construire un mode RAG dans l'application existante pour que l'assistant puisse:
- importer des documents (`.pdf`, `.txt`, `.md`),
- indexer leur contenu,
- répondre avec contexte documentaire,
- gérer les documents (ajout, liste, suppression) depuis l'UI.

## 2) Résultat attendu (MVP)
- Upload de fichiers depuis l'UI.
- Stockage local des fichiers et index vectoriel persistant.
- Endpoint backend de recherche contextuelle + génération de réponse avec citations.
- Liste des documents indexés dans l'UI avec suppression.
- Toggle "Mode RAG" dans le chat (on/off).

## 3) Contraintes et choix techniques (proposés)
- Embeddings locaux via Ollama embeddings (ou modèle sentence-transformers local en fallback).
- Vector store local: Chroma persisté sur disque.
- Parsing:
  - PDF: `pypdf` (MVP),
  - TXT/MD: lecture UTF-8.
- Chunking: `RecursiveCharacterTextSplitter`.
- Métadonnées minimales: `doc_id`, `filename`, `type`, `chunk_id`, `created_at`.

## 4) Plan par phases

### Phase 0 — Préparation
- [ ] Créer l'arborescence `data/uploads` et `data/vectorstore`.
- [ ] Définir une convention d'identifiants documents (`doc_id`).
- [ ] Ajouter un fichier de config central (`RAG_ENABLED`, chemins, taille chunk, overlap, top_k).

### Phase 1 — Backend Ingestion
- [ ] Endpoint `POST /rag/documents` (upload multi-fichiers).
- [ ] Validation extensions (`.pdf`, `.txt`, `.md`) + taille max fichier.
- [ ] Extraction texte par type.
- [ ] Chunking + embeddings + upsert dans le vector store.
- [ ] Retour API: documents ajoutés + erreurs détaillées par fichier.

### Phase 2 — Backend Gestion documents
- [ ] Endpoint `GET /rag/documents` (liste documents indexés).
- [ ] Endpoint `DELETE /rag/documents/:doc_id` (supprimer index + fichier local).
- [ ] Endpoint `POST /rag/reindex` (optionnel MVP+, pour reconstruction complète).

### Phase 3 — Backend Question/Réponse RAG
- [ ] Endpoint `POST /rag/chat` (question + options RAG).
- [ ] Retrieval `top_k` + score.
- [ ] Construction d'un prompt de réponse basé sur les chunks récupérés.
- [ ] Réponse JSON: `reply`, `sources` (nom fichier + extrait + score).
- [ ] Garde-fou: si aucun contexte pertinent, répondre explicitement sans halluciner.

### Phase 4 — Intégration dans `/chat` existant
- [ ] Ajouter option `use_rag` côté requête chat.
- [ ] Si `use_rag=true`, router vers pipeline RAG avant réponse finale.
- [ ] Conserver les outils existants (heure, note) sans régression.

### Phase 5 — UI Gestion des fichiers
- [ ] Créer un panneau "Documents" avec:
  - [ ] bouton Upload,
  - [ ] liste documents,
  - [ ] suppression document,
  - [ ] indicateur d'indexation (succès/erreur).
- [ ] Afficher nombre de documents et dernière mise à jour.
- [ ] Gérer erreurs UX (format non supporté, fichier vide, extraction impossible).

### Phase 6 — UI Chat RAG
- [ ] Ajouter toggle "Mode RAG" dans le header/chat.
- [ ] Envoyer `use_rag` au backend.
- [ ] Afficher les sources sous la réponse (fichier + extrait court).
- [ ] Ajouter état de chargement spécifique "Analyse des documents...".

### Phase 7 — Qualité / Tests
- [ ] Tests unitaires extraction texte (`pdf/txt/md`).
- [ ] Tests unitaires de chunking et métadonnées.
- [ ] Tests API (upload/list/delete/rag chat).
- [ ] Test manuel end-to-end UI + backend.
- [ ] Vérifier persistance après redémarrage serveur.

### Phase 8 — Durcissement (après MVP)
- [ ] Filtrage par document sélectionné dans l'UI.
- [ ] Réindexation incrémentale plus robuste.
- [ ] Déduplication hash fichiers.
- [ ] Support `docx/csv`.
- [ ] Streaming des réponses RAG.

## 5) Définition de terminé (MVP DoD)
Le MVP est terminé si:
- un utilisateur peut importer plusieurs documents (`pdf/txt/md`),
- ils apparaissent dans la liste UI,
- il peut en supprimer,
- en mode RAG, le chat répond en se basant sur ces documents,
- et chaque réponse affiche ses sources.

## 6) Risques à surveiller
- Qualité d'extraction PDF (documents scannés non OCR).
- Lenteur à l'indexation sur gros fichiers.
- Hallucinations si prompt RAG mal contraint.
- Cohérence suppression fichier + suppression vecteurs.

## 7) Ordre d'exécution recommandé (pratique)
1. Backend ingestion + listing + suppression.
2. Endpoint `rag/chat` avec citations.
3. UI Documents (upload/list/delete).
4. Toggle RAG dans le chat.
5. Tests + ajustements performance.

## 8) Première tâche concrète à lancer
- Implémenter `POST /rag/documents` + extraction texte + indexation Chroma persistée.
