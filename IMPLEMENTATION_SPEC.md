# The Living Kernel — Spec d'implémentation & Liste de tâches

> Basé sur les 3 livrables (CdCF, CdCT, Backlog US) et l'état actuel du projet.

---

## État du projet actuel

### Déjà implémenté ✅

| Composant | Ce qui existe |
|-----------|--------------|
| **Backend Flask** | `server.py` — API REST, agent ReAct via LangChain + Ollama |
| **Tools basiques** | `langchain_tools.py` — `get_current_time`, `create_note` |
| **Confirmation d'outil** | Middleware de confirmation avant exécution des outils sensibles |
| **RAG Pipeline** | Upload documents (PDF/TXT/MD), chunking, embeddings Ollama, vectorstore ChromaDB local |
| **Frontend React** | Interface chat avec threads, outil de confirmation, affichage sources RAG, gestion documents |
| **Tests** | `tests/test_rag_api.py`, `tests/test_rag_unit.py` |
| **Vectorstore local** | `data/vectorstore/` + `data/documents_registry.json` |

### Ce qui manque (périmètre de ce spec) ❌

- Architecture LangGraph (nœuds PLANNER / FORGE / TESTER / EXECUTOR / GOVERNOR)
- Système de missions (UUID, statut, contexte, niveau d'autonomie)
- Forge automatique d'outils Python
- Système de skills (brain/skills/*.md)
- Sandbox d'exécution isolée
- Registre officiel des outils (versionnement, statuts)
- Base de données structurée (PostgreSQL ou SQLite en dev)
- Console d'administration
- Journal d'audit et historique des missions
- Couche de gouvernance (permissions, politiques, validation humaine)

---

## Architecture cible

```
┌─────────────────────────────────────────────────────┐
│                   Frontend React/TS                  │
│  Chat | Missions | Skills | Tools | Admin | Logs    │
└────────────────────┬────────────────────────────────┘
                     │ HTTP / REST
┌────────────────────▼────────────────────────────────┐
│              API FastAPI (remplace Flask)            │
│  /missions  /tools  /skills  /admin  /rag  /chat    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│            Moteur LangGraph (core/)                  │
│                                                     │
│  ┌─────────┐  ┌───────┐  ┌────────┐  ┌──────────┐ │
│  │ PLANNER │→ │ FORGE │→ │ TESTER │→ │ EXECUTOR │ │
│  └────┬────┘  └───────┘  └────────┘  └──────────┘ │
│       │              ↑ feedback loop               │
│  ┌────▼──────┐  ┌───────────┐  ┌────────────────┐ │
│  │ GOVERNOR  │  │  LOGGER   │  │    REGISTRY    │ │
│  └───────────┘  └───────────┘  └────────────────┘ │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                   Persistance                        │
│  PostgreSQL/SQLite | ChromaDB | Filesystem local    │
│  /brain/skills/    /tools/    /sandbox/  /logs/     │
└─────────────────────────────────────────────────────┘
```

---

## Modules à implémenter

### Module A — Mission Manager

**Objectif :** Gérer le cycle de vie des missions utilisateur.

**Modèle de données :**
```python
class Mission:
    id: UUID
    title: str
    prompt: str
    context_path: str | None
    autonomy_level: Literal["restricted", "supervised", "extended"]
    status: Literal["pending", "planning", "forging", "testing", "executing", "done", "error"]
    risk_level: str | None
    started_at: datetime
    ended_at: datetime | None

class MissionStep:
    id: UUID
    mission_id: UUID
    node_name: str   # PLANNER, FORGE, TESTER, EXECUTOR
    status: str
    input_payload: dict
    output_payload: dict | None
    error_payload: dict | None
    started_at: datetime
    ended_at: datetime | None
```

**Endpoints API :**
- `POST /missions` — créer une mission
- `GET /missions` — liste (filtrable par statut)
- `GET /missions/{id}` — détail + steps
- `GET /missions/{id}/steps` — traçabilité du chemin décisionnel

---

### Module B — Planner Engine (nœud PLANNER)

**Objectif :** Analyser la mission, chercher une skill existante, décider du routing.

**Logique :**
1. Recevoir la mission
2. Interroger `brain/skills/` + vectorstore pour trouver une skill applicable
3. Si skill trouvée → router vers EXECUTOR
4. Si non → router vers FORGE
5. Logger la décision

**Interface :**
```python
class PlannerOutput(BaseModel):
    decision: Literal["use_existing_skill", "forge_new_tool"]
    skill_id: str | None
    rationale: str
    risk_level: str
```

---

### Module C — Forge Engine (nœud FORGE)

**Objectif :** Générer dynamiquement un outil Python et une skill Markdown.

**Template d'outil généré :**
```python
# tool_{slug}.py
# id: {uuid}
# version: 1.0.0
# created_by_mission: {mission_id}

"""
{description}
Usage: {usage}
Limits: {limits}
"""

def run(args: dict) -> dict:
    """Point d'entrée standard."""
    ...
```

**Template de skill générée :**
```markdown
---
id: {uuid}
title: {title}
tool_slug: {slug}
status: candidate
created_at: {iso_date}
---
# {title}
## Contexte d'usage
## Prérequis
## Mode d'appel
## Limites connues
```

**Boucle de correction :** si TESTER renvoie une erreur, FORGE reçoit le traceback et tente une correction (max configurable, défaut : 3 tentatives).

---

### Module D — Test Engine (nœud TESTER)

**Objectif :** Exécuter l'outil candidat dans un environnement isolé et capturer les résultats.

**Sandbox :** subprocess Python avec restrictions (timeout, pas d'accès réseau, pas d'écriture hors `/sandbox/`).

**Modèle :**
```python
class ToolTestRun:
    id: UUID
    tool_id: UUID
    mission_id: UUID
    attempt_number: int
    status: Literal["success", "failure"]
    stdout: str
    stderr: str
    traceback: str | None
    created_at: datetime
```

---

### Module E — Registry & Memory (nœud REGISTRY)

**Objectif :** Enregistrer les outils validés, gérer le cycle de vie des skills.

**Modèles :**
```python
class Tool:
    id: UUID
    name: str
    slug: str        # unique
    version: str
    description: str
    file_path: str
    status: Literal["candidate", "active", "disabled", "archived"]
    created_by_mission_id: UUID
    created_at: datetime

class Skill:
    id: UUID
    tool_id: UUID | None
    title: str
    slug: str
    summary: str
    markdown_path: str
    status: Literal["candidate", "active", "disabled"]
    created_at: datetime

class RegistryEntry:
    id: UUID
    tool_id: UUID
    skill_id: UUID
    published_version: str
    validation_status: Literal["pending", "approved", "rejected"]
    approved_by: str | None
    published_at: datetime | None
```

---

### Module F — Governance Layer (nœud GOVERNOR)

**Objectif :** Vérifier les permissions, intercepter les actions risquées, gérer la validation humaine.

**Politiques :**
- `autonomy_level = restricted` → validation humaine obligatoire avant toute forge
- `autonomy_level = supervised` → validation avant enregistrement au registre
- `autonomy_level = extended` → exécution complète sans blocage humain
- Blacklist d'actions toujours interdites (ex: `rm -rf`, accès réseau externe)
- Max tentatives de correction configurable

**Endpoint :** `POST /missions/{id}/approve` — validation humaine d'un outil candidat

---

### Module G — Admin Console (frontend)

**Objectif :** Interface de supervision des outils, skills, missions, logs.

**Vues à ajouter au frontend existant :**
- `/admin/missions` — historique filtrable
- `/admin/tools` — liste avec statuts, actions (désactiver, archiver)
- `/admin/skills` — bibliothèque de compétences
- `/admin/logs` — logs par mission avec tracebacks
- `/admin/registry` — registre officiel avec validation

---

## Stack technique

| Couche | Choix |
|--------|-------|
| Backend | **FastAPI** (migration depuis Flask) |
| Moteur agentique | **LangGraph** (graphe d'état) |
| LLM local | **Ollama** (déjà en place) |
| Base de données | **SQLite** en dev → PostgreSQL en prod |
| ORM | **SQLAlchemy** + **Alembic** (migrations) |
| Vectorstore | **ChromaDB** (déjà en place) |
| Sandbox | **subprocess** restreint + éventuellement Docker |
| Frontend | React/TS (déjà en place) — ajout de vues admin |
| Tests | pytest (déjà en place) |

---

## Liste de tâches

### Sprint 1 — Fondations & Migration backend

- [ ] **T1** Migrer `server.py` de Flask vers FastAPI (conserver toutes les routes existantes)
- [ ] **T2** Mettre en place SQLAlchemy + SQLite avec les modèles `Mission`, `MissionStep`, `Tool`, `Skill`, `ToolTestRun`, `RegistryEntry`, `AuditLog`
- [ ] **T3** Créer les migrations Alembic initiales
- [ ] **T4** Implémenter les endpoints CRUD `/missions` (create, list, get, steps)
- [ ] **T5** Créer la structure de fichiers `brain/`, `brain/skills/`, `tools/`, `sandbox/`, `logs/`, `artifacts/`
- [ ] **T6** Migrer le pipeline RAG existant dans FastAPI sans régression

---

### Sprint 2 — Moteur LangGraph (nœuds principaux)

- [ ] **T7** Installer LangGraph et créer le squelette du graphe dans `core/graph/`
- [ ] **T8** Implémenter le nœud **PLANNER** : analyse mission, recherche skill vectorielle, décision routing
- [ ] **T9** Implémenter le nœud **EXECUTOR** : exécuter la skill existante, retourner le résultat
- [ ] **T10** Implémenter le nœud **LOGGER** : persister chaque `MissionStep` en BDD à chaque transition
- [ ] **T11** Connecter le graphe à l'endpoint `POST /missions/{id}/run`

---

### Sprint 3 — Forge & Test Engine

- [ ] **T12** Implémenter le nœud **FORGE** : génération de code Python + skill Markdown via LLM
- [ ] **T13** Créer le template standard d'outil forgé (interface `run(args) -> dict`, docstring, gestion d'erreurs)
- [ ] **T14** Implémenter le nœud **TESTER** : exécution subprocess isolé avec timeout et capture stdout/stderr/traceback
- [ ] **T15** Implémenter la boucle de correction FORGE ↔ TESTER (max_attempts configurable, défaut 3)
- [ ] **T16** Persister les `ToolTestRun` en BDD

---

### Sprint 4 — Registre & Mémoire

- [ ] **T17** Implémenter le nœud **REGISTRY** : enregistrement des outils validés avec versionning
- [ ] **T18** Endpoints `/tools` (list, get, update status) et `/skills` (list, get, update status)
- [ ] **T19** Indexer automatiquement les nouvelles skills dans ChromaDB à la validation
- [ ] **T20** Implémenter l'archivage / désactivation d'outils (status lifecycle)
- [ ] **T21** Créer `brain/agent.md` et le maintenir à jour (identité, capacités actuelles)

---

### Sprint 5 — Gouvernance & Sécurité

- [ ] **T22** Implémenter le nœud **GOVERNOR** : vérification des politiques selon `autonomy_level`
- [ ] **T23** Endpoint `POST /missions/{id}/approve` — validation humaine d'un outil candidat
- [ ] **T24** Implémenter `AuditLog` — journalisation systématique de toutes les actions critiques
- [ ] **T25** Configurer la blacklist d'actions interdites (commandes shell destructrices, accès réseau contrôlé)
- [ ] **T26** Ajouter le rate limiting sur l'API (ex: `slowapi`)

---

### Sprint 6 — Console d'administration (frontend)

- [ ] **T27** Créer la vue `/admin/missions` — liste filtrée avec statuts et détail des steps
- [ ] **T28** Créer la vue `/admin/tools` — liste avec actions (désactiver, archiver, voir tests)
- [ ] **T29** Créer la vue `/admin/skills` — bibliothèque avec fiche détail
- [ ] **T30** Créer la vue `/admin/logs` — logs par mission avec tracebacks
- [ ] **T31** Créer la vue `/admin/registry` — outils en attente de validation avec bouton approve/reject
- [ ] **T32** Intégrer une navbar et le routing entre les vues existantes (Chat, RAG) et les nouvelles vues Admin

---

### Sprint 7 — Tests & Observabilité

- [ ] **T33** Tests unitaires des nœuds LangGraph (PLANNER, FORGE, TESTER, EXECUTOR)
- [ ] **T34** Tests d'intégration : mission complète avec skill existante (flow 1)
- [ ] **T35** Tests d'intégration : mission complète avec forge + validation (flow 2)
- [ ] **T36** Tests d'intégration : mission en échec avec boucle de correction (flow 3)
- [ ] **T37** Tests de sécurité sandbox : tentative commande interdite, dépassement timeout
- [ ] **T38** Exposer les métriques clés : taux de succès, nombre moyen de boucles, temps de forge

---

## Ordre de priorité MVP (MoSCoW)

### Must Have
- T1 à T11 (migration FastAPI + graphe LangGraph avec PLANNER + EXECUTOR)
- T12 à T16 (FORGE + TESTER + boucle de correction)
- T17, T18 (REGISTRY basique)
- T22, T23, T24 (GOVERNOR + validation humaine + audit)

### Should Have
- T19 à T21 (indexation skills, archivage, agent.md)
- T25, T26 (sécurité sandbox, rate limiting)
- T27 à T31 (vues admin)

### Could Have
- T32 (routing/navbar frontend)
- T33 à T38 (tests complets + métriques)

### Won't Have (MVP)
- Marketplace publique de skills
- Multi-tenant
- Exécution distribuée
- Fine-tuning

---

## Conventions de développement

- Chaque outil forgé **doit** exposer `run(args: dict) -> dict`
- Chaque skill **doit** avoir un frontmatter YAML (id, title, tool_slug, status, created_at)
- Les nœuds LangGraph **doivent** persister leur `MissionStep` avant de passer au suivant
- Aucun outil non validé ne peut être appelé hors sandbox
- Tout `AuditLog` inclut : `actor_type`, `action`, `target_type`, `target_id`, `created_at`
