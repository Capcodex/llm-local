# TASK.md — The Living Kernel

> Suivi des tâches d'implémentation. Voir [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md) pour le détail technique de chaque module.

---

## Sprint 1 — Fondations & Migration backend

- [x] **T1** Migrer `server.py` de Flask vers FastAPI
  - Conserver toutes les routes existantes (`/chat`, `/rag/*`)
  - Remplacer `flask`, `flask_cors` par `fastapi`, `uvicorn`, `python-multipart`
  - Adapter la gestion des requêtes et réponses (Pydantic models)

- [x] **T2** Mettre en place SQLAlchemy + SQLite avec les modèles de données
  - `Mission`, `MissionStep`, `Tool`, `Skill`, `ToolTestRun`, `RegistryEntry`, `AuditLog`
  - Fichier `core/models.py`

- [x] **T3** Créer les migrations Alembic initiales
  - Initialiser Alembic (`alembic init`)
  - Générer la migration pour tous les modèles de T2

- [x] **T4** Implémenter les endpoints CRUD `/missions`
  - `POST /missions` — créer une mission (prompt, context_path, autonomy_level)
  - `GET /missions` — liste filtrée par statut
  - `GET /missions/{id}` — détail mission
  - `GET /missions/{id}/steps` — traçabilité du chemin décisionnel

- [x] **T5** Créer la structure de fichiers projet
  - `brain/`, `brain/skills/`, `tools/`, `sandbox/`, `logs/`, `artifacts/`
  - Fichier `brain/agent.md` (identité initiale de l'agent)

- [x] **T6** Migrer le pipeline RAG existant dans FastAPI sans régression
  - Vérifier que tous les tests existants passent après migration

---

## Sprint 2 — Moteur LangGraph (nœuds principaux)

- [x] **T7** Installer LangGraph et créer le squelette du graphe
  - `pip install langgraph`
  - Créer `core/graph/graph.py` avec l'état partagé (`GraphState`)
  - Définir les nœuds et transitions de base

- [x] **T8** Implémenter le nœud **PLANNER**
  - Analyser le prompt de la mission via LLM (Ollama)
  - Rechercher une skill correspondante dans ChromaDB + `brain/skills/`
  - Retourner `decision: use_existing_skill | forge_new_tool` + justification
  - Persister le `MissionStep` PLANNER en BDD

- [x] **T9** Implémenter le nœud **EXECUTOR**
  - Charger et exécuter la skill existante identifiée par PLANNER
  - Formater le résultat final pour l'utilisateur
  - Persister le `MissionStep` EXECUTOR en BDD

- [x] **T10** Implémenter le nœud **LOGGER**
  - Centraliser la persistance de chaque transition dans `MissionStep`
  - Écrire les logs dans `logs/` par mission_id

- [x] **T11** Connecter le graphe à l'API
  - Endpoint `POST /missions/{id}/run` — déclenche l'exécution du graphe
  - Retourner le résultat final et l'id des steps

---

## Sprint 3 — Forge & Test Engine

- [x] **T12** Implémenter le nœud **FORGE** — génération de l'outil
  - Prompt LLM pour générer un fichier Python respectant le template standard
  - Sauvegarder dans `tools/tool_{slug}.py`
  - Persister l'entrée `Tool` (statut `candidate`) en BDD

- [x] **T13** Implémenter le nœud **FORGE** — génération de la skill
  - Prompt LLM pour générer le fichier Markdown de skill
  - Sauvegarder dans `brain/skills/skill_{slug}.md`
  - Persister l'entrée `Skill` (statut `candidate`) en BDD

- [x] **T14** Implémenter le nœud **TESTER** — sandbox d'exécution
  - Lancer l'outil candidat dans un subprocess Python isolé
  - Appliquer un timeout configurable (défaut : 10s)
  - Capturer `stdout`, `stderr`, `traceback`
  - Persister le `ToolTestRun` en BDD

- [x] **T15** Implémenter la boucle de correction FORGE ↔ TESTER
  - Si TESTER échoue → renvoyer traceback à FORGE pour correction
  - Limite configurable de tentatives (défaut : 3, `MAX_FORGE_ATTEMPTS`)
  - Au-delà → passer la mission en statut `error` avec motif d'arrêt

- [x] **T16** Vérifier la persistance complète des `ToolTestRun`
  - Chaque tentative est numérotée (`attempt_number`)
  - Le dernier statut est propagé vers l'outil parent

---

## Sprint 4 — Registre & Mémoire

- [x] **T17** Implémenter le nœud **REGISTRY** — enregistrement des outils validés
  - Créer l'entrée `RegistryEntry` après succès TESTER
  - Versionner l'outil (`1.0.0`, puis incrémentation automatique)
  - Mettre le statut `Tool` à `active`

- [x] **T18** Endpoints `/tools` et `/skills`
  - `GET /tools` — liste avec filtres statut
  - `GET /tools/{id}` — détail + historique de tests
  - `PATCH /tools/{id}` — changer le statut (disable, archive)
  - `GET /skills` — liste
  - `GET /skills/{id}` — fiche détail Markdown

- [x] **T19** Indexer les nouvelles skills dans ChromaDB à la validation
  - Après enregistrement d'une skill active → upsert dans le vectorstore

- [x] **T20** Implémenter l'archivage / désactivation d'outils
  - Un outil `disabled` n'est plus proposé par PLANNER
  - L'historique reste consultable

- [x] **T21** Maintenir `brain/agent.md` à jour
  - Ajouter une ligne de résumé après chaque nouveau skill enregistré

---

## Sprint 5 — Gouvernance & Sécurité

- [x] **T22** Implémenter le nœud **GOVERNOR**
  - Vérifier `autonomy_level` de la mission avant chaque action sensible
  - `restricted` → bloquer jusqu'à validation humaine avant forge
  - `supervised` → bloquer avant enregistrement au registre
  - `extended` → laisser passer

- [x] **T23** Endpoint de validation humaine
  - `POST /missions/{id}/approve` — approuver un outil candidat
  - `POST /missions/{id}/reject` — rejeter avec motif
  - Persister la décision dans `RegistryEntry.validation_status` et `AuditLog`

- [x] **T24** Implémenter `AuditLog` systématique
  - Logger toutes les actions critiques : forge, test, enregistrement, approbation, suppression
  - Champs : `actor_type`, `action`, `target_type`, `target_id`, `metadata`, `created_at`

- [x] **T25** Sécuriser la sandbox d'exécution
  - Blacklist de patterns interdits dans le code généré (`os.system`, `subprocess`, `shutil.rmtree`, etc.)
  - Accès système de fichiers restreint au dossier `sandbox/` uniquement
  - Pas d'accès réseau (bloquer les imports `requests`, `urllib` dans le code candidat)

- [x] **T26** Ajouter le rate limiting sur l'API
  - Installer `slowapi`
  - Limiter `/missions` à N requêtes/minute par IP

---

## Sprint 6 — Console d'administration (frontend)

- [ ] **T27** Vue `/admin/missions` — historique des missions
  - Liste filtrée par statut (pending, done, error)
  - Détail avec steps, outil utilisé, résultat

- [ ] **T28** Vue `/admin/tools` — bibliothèque d'outils
  - Liste avec statuts
  - Actions : désactiver, archiver
  - Détail : code source, historique de tests, skill associée

- [ ] **T29** Vue `/admin/skills` — bibliothèque de compétences
  - Liste avec titre, résumé, statut
  - Fiche détail rendu Markdown

- [ ] **T30** Vue `/admin/logs` — journaux techniques
  - Logs par mission_id
  - Tracebacks visibles
  - Filtre par statut

- [ ] **T31** Vue `/admin/registry` — validation des outils candidats
  - Liste des outils en attente (`validation_status: pending`)
  - Boutons Approuver / Rejeter

- [ ] **T32** Navigation et routing frontend
  - Ajouter une navbar avec les sections : Chat | Documents | Missions | Admin
  - Routing React entre les vues existantes et nouvelles

---

## Sprint 7 — Tests & Observabilité

- [ ] **T33** Tests unitaires des nœuds LangGraph
  - PLANNER : routing correct selon skill présente / absente
  - FORGE : code généré respecte le template standard
  - TESTER : capture correcte stdout/stderr/traceback
  - EXECUTOR : résultat correctement formaté

- [ ] **T34** Test d'intégration — Flow 1 : skill existante
  - Mission → PLANNER trouve skill → EXECUTOR → résultat OK

- [ ] **T35** Test d'intégration — Flow 2 : forge + validation
  - Mission → PLANNER ne trouve rien → FORGE → TESTER succès → REGISTRY → EXECUTOR

- [ ] **T36** Test d'intégration — Flow 3 : boucle de correction
  - Mission → FORGE génère code invalide → TESTER échoue → FORGE corrige → TESTER succès

- [ ] **T37** Tests de sécurité sandbox
  - Tentative d'exécution de commande interdite → rejet
  - Dépassement timeout → arrêt propre
  - Code tentant d'écrire hors `sandbox/` → rejet

- [ ] **T38** Métriques et observabilité
  - Endpoint `GET /metrics` : taux de succès missions, nombre moyen de boucles FORGE, temps moyen de forge
  - Taux de réutilisation skill vs création

---

## Backlog (post-MVP)

- [ ] Sandbox Docker (isolation renforcée)
- [ ] Support PostgreSQL en production
- [ ] Streaming des réponses longues (SSE)
- [ ] Filtrage RAG par document sélectionné dans l'UI
- [ ] Réindexation incrémentale + déduplication hash
- [ ] Support `docx`, `csv` dans le pipeline RAG
- [ ] Notifications webhook (Slack / email) sur validation requise
- [ ] Export des logs et artefacts
- [ ] Multi-modèles (sélection du modèle par mission)

---

## Définition de terminé (DoD MVP)

Le MVP est terminé si :
- Un utilisateur peut soumettre une mission textuelle avec un niveau d'autonomie
- L'agent cherche une skill existante avant de forger
- Si aucune skill → un outil Python est généré, testé et enregistré automatiquement
- En cas d'erreur de test → l'agent tente une correction (max 3 fois)
- Le résultat final est retourné avec le chemin décisionnel complet
- Un administrateur peut consulter les missions, outils, skills et logs
- Les outils sensibles peuvent être soumis à validation humaine avant enregistrement
