-- Publishing database schema
-- Requires: PostgreSQL with pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;

-- Sources (channels, repos)
CREATE TABLE sources (
    id SERIAL PRIMARY KEY,
    type VARCHAR(20) NOT NULL,  -- 'youtube_channel', 'github_repo'
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL UNIQUE,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Content items (videos, files)
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES sources(id),
    type VARCHAR(20) NOT NULL,  -- 'video', 'code_file'
    title VARCHAR(500),
    url TEXT UNIQUE,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chunks (searchable segments with embeddings)
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER,
    timestamp_start FLOAT,  -- seconds, for video chunks
    timestamp_end FLOAT,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create vector similarity index
-- Note: For better recall, rebuild this after adding data:
--   DROP INDEX chunks_embedding_idx;
--   CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

-- Skills (presentation skills generated from research)
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL,  -- 'diagnostic', 'tutorial', 'comparison', etc.
    description TEXT,
    content TEXT NOT NULL,  -- the skill instructions/template
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Projects (organize research, skills, and outputs)
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active',  -- 'active', 'completed', 'archived'

    -- Three facets
    facet_about JSONB DEFAULT '[]',
    facet_uses JSONB DEFAULT '[]',
    facet_needs JSONB DEFAULT '[]',

    -- Embeddings for semantic matching (cross-pollination)
    embedding_about vector(1536),
    embedding_uses vector(1536),
    embedding_needs vector(1536),

    -- Link to beads molecule
    beads_epic_id VARCHAR(50),

    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Link items to projects
CREATE TABLE project_items (
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    item_id INTEGER REFERENCES items(id) ON DELETE CASCADE,
    PRIMARY KEY (project_id, item_id)
);

-- Link skills to projects
CREATE TABLE project_skills (
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    skill_id INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    PRIMARY KEY (project_id, skill_id)
);

-- Project outputs (generated deliverables)
CREATE TABLE project_outputs (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50),  -- 'pdf', 'markdown', 'image', etc.
    path TEXT,  -- file path to output
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
