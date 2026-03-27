# LangChain RAG Chat

Application de chat local avec **Retrieval-Augmented Generation (RAG)** — posez des questions sur vos propres documents (PDF, TXT, Markdown) grâce à un LLM tournant entièrement en local via Ollama.

## Aperçu

| Composant | Technologie |
|-----------|-------------|
| Backend | Python / Flask / LangChain |
| Frontend | React 19 / TypeScript / Vite |
| LLM | Ollama (`qwen3:latest`) |
| Embeddings | Ollama (`nomic-embed-text:latest`) |
| Base vectorielle | Chroma (persistant) |

**Fonctionnalités :**
- Chat multi-conversations (persisté dans le navigateur)
- Upload de documents PDF, TXT, Markdown
- Mode RAG : les réponses citent les passages sources
- Outils LangChain : heure actuelle, création de notes (avec confirmation)
- Tests unitaires et d'intégration (pytest)

---

## Prérequis

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) installé et en cours d'exécution

---

## 1. Lancer Ollama et télécharger les modèles

Ollama doit tourner avant de démarrer le backend.

```bash
# Démarrer le service Ollama (si pas déjà lancé)
ollama serve
```

Dans un autre terminal, téléchargez les deux modèles nécessaires :

```bash
# Modèle de langage (chat)
ollama pull qwen3:latest

# Modèle d'embeddings (vectorisation des documents)
ollama pull nomic-embed-text:latest
```

Vérifiez que les modèles sont disponibles :

```bash
ollama list
```

---

## 2. Installer les dépendances Python

À la racine du projet :

```bash
pip install flask flask-cors langchain langchain-community langchain-text-splitters chromadb pypdf werkzeug
```

---

## 3. Lancer le backend Flask

```bash
# Depuis la racine du projet
python server.py
```

Le serveur démarre sur **http://localhost:5001**.

---

## 4. Lancer le frontend

```bash
cd ollama-chat-ui

# Installer les dépendances Node (première fois uniquement)
npm install

# Démarrer en mode développement
npm run dev
```

L'interface est accessible sur **http://localhost:5173**.

---

## 5. Utiliser l'application

1. Ouvrez http://localhost:5173 dans votre navigateur
2. **Chat simple** : tapez un message et envoyez
3. **Mode RAG** :
   - Cliquez sur l'icône de gestion de documents (panneau latéral)
   - Uploadez un fichier PDF, TXT ou Markdown (max 10 Mo)
   - Activez le toggle "RAG" dans l'interface de chat
   - Posez une question — la réponse citera les passages pertinents

---

## Structure du projet

```
.
├── server.py               # Backend Flask + pipeline RAG
├── langchain_tools.py      # Outils LangChain (heure, création de notes)
├── data/
│   ├── uploads/            # Fichiers uploadés
│   ├── vectorstore/        # Base vectorielle Chroma (persistante)
│   └── documents_registry.json
├── ollama-chat-ui/
│   ├── src/
│   │   ├── App.tsx         # Composant React principal
│   │   └── index.css       # Styles
│   ├── dist/               # Build de production
│   └── package.json
└── tests/
    ├── conftest.py
    ├── test_rag_unit.py
    └── test_rag_api.py
```

---

## Lancer les tests

```bash
pytest tests/
```

---

## Build de production (frontend)

```bash
cd ollama-chat-ui
npm run build
# Les fichiers sont générés dans ollama-chat-ui/dist/
```

---

## Configuration

Les paramètres principaux se trouvent en haut de [server.py](server.py) :

| Variable | Valeur par défaut | Description |
|----------|-------------------|-------------|
| `DEFAULT_MODEL_NAME` | `qwen3:latest` | Modèle LLM Ollama |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text:latest` | Modèle d'embeddings |
| `RAG_CHUNK_SIZE` | `1200` | Taille des chunks de texte |
| `RAG_CHUNK_OVERLAP` | `200` | Chevauchement entre chunks |
| `RAG_DEFAULT_TOP_K` | `4` | Nombre de chunks récupérés |
| `RAG_MAX_FILE_SIZE_BYTES` | `10 Mo` | Taille maximale des fichiers |
| `MAX_CONTEXT_MESSAGES` | `12` | Fenêtre de contexte du chat |

---

## API Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/rag/documents` | Uploader un document |
| `GET` | `/rag/documents` | Lister les documents indexés |
| `DELETE` | `/rag/documents/<doc_id>` | Supprimer un document |
| `POST` | `/rag/chat` | Chat avec RAG |
| `POST` | `/chat` | Chat général avec outils LangChain |
