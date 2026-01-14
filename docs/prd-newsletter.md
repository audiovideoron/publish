# Inspire Record
## Product Requirements Document

**Company:** Inspire Solutions, Dallas, TX
**Product:** Internal knowledge base and operational assistant
**Version:** 1.0 MVP
**Date:** January 2026

---

## 1. Executive Summary

Inspire Record is an operational system that does things instead of explaining how to do things.

The current SharePoint site is a library of documentation. Employees read procedures, hunt for templates, and manually execute multi-step workflows. Inspire Record replaces this with an assistant that acts: when an employee needs to cross-rent equipment, the system presents the form, routes the submission, and tracks the status — all searchable through the same semantic interface.

The core insight: **content should be the interface, not a description of the interface.**

All content — policies, training videos, forms, submitted requests — flows into a unified pgvector knowledge base. Semantic search retrieves both reference material ("What's the PTO policy?") and operational records ("What's the status of my equipment request?").

---

## 2. Problem Statement

**Current State:**
- Documentation explains procedures instead of executing them
- Employees read "email this person, download this form, attach it here" — then do it manually
- Policy updates arrive via email, then change a month later
- Multiple versions exist across SharePoint, email threads, and Slack
- No authoritative source when conflicts arise
- Keyword search requires knowing exact terminology
- Forms and templates scattered across file shares
- No way to check request status without emailing someone
- Institutional knowledge leaves when employees leave

**Desired State:**
- System executes workflows, not just describes them
- Forms embedded directly — employees fill them out, system routes them
- Submitted requests become searchable records
- Semantic search: ask questions in plain language, get answers and actions
- Single source of truth with versioning and supersession
- Unified index: policies, training, forms, and transaction records in one searchable space

---

## 3. Content Types

### 3.1 Reference Content (Read)

| Type | Purpose | Example |
|------|---------|---------|
| **Procedure** | SOPs, policies, how-tos | "Cross-Rental Equipment Policy" |
| **Story** | Problem/solution narratives | "How Dallas Cut Setup Time by 40%" |
| **Spotlight** | Division/team achievements | "Q4 Wins from the AV Team" |
| **Announcement** | Policy changes, new tools | "Updated Travel Reimbursement Policy" |
| **Tutorial** | Learning/instruction | "Using AI to Draft Client Proposals" |
| **Training** | Harvested video transcripts | Radar Essentials course |

### 3.2 Operational Content (Do)

| Type | Purpose | Example |
|------|---------|---------|
| **Form** | Actionable templates | Cross-Rental Request, PTO Request |
| **Submission** | Completed form records | "Houston projector request - March 15" |
| **Workflow** | Multi-step processes | PO approval chain |

Forms are embedded directly in procedures. When the system retrieves "How do I cross-rent equipment?", it returns the policy AND presents the form to fill out. Submissions become searchable records.

---

## 4. Information Architecture

### 4.1 Content Hierarchy

```
Knowledge Base (pgvector)
│
├── Reference Layer (Read)
│   ├── Articles (policies, procedures, announcements)
│   │   ├── metadata: type, topics, division, author
│   │   ├── versioning: supersedes, is_current
│   │   └── embedded_form_id (optional link to actionable form)
│   │
│   ├── Training (harvested video transcripts)
│   │   └── chunked and embedded for semantic search
│   │
│   └── Issues (periodic publications)
│       └── contains 1-N articles
│
├── Operational Layer (Do)
│   ├── Forms (templates)
│   │   ├── fields: name, type, required, options
│   │   ├── routing: who receives submissions
│   │   └── linked from procedures
│   │
│   └── Submissions (completed forms = records)
│       ├── form_id, submitter, timestamp
│       ├── field_values (JSONB)
│       ├── status: pending, approved, rejected, completed
│       └── embedded for semantic search
│
└── All content embedded → unified semantic search
```

### 4.2 Organizational Structure (Future)

```
Inspire Solutions
├── Hospitality Division
│   ├── Regions (Southeast, Southwest, ...)
│   │   ├── RVP (Regional Vice President)
│   │   └── Properties
│   │       └── DET (Director of Event Technology)
└── Other Divisions
```

---

## 5. User Workflows

### 5.1 Semantic Query (Primary Interface)

Employee asks a question in plain language. System responds with action, not just information.

**Example: "I need to rent a projector from Houston for March 15"**

1. System searches embedded content for relevant procedures
2. Returns: Cross-Rental Equipment Policy (summary)
3. Presents: Cross-Rental Request form (pre-populated where possible)
4. Employee fills remaining fields, submits
5. System routes to approver, creates searchable submission record
6. Employee can later ask: "What's the status of my Houston request?"

### 5.2 Form Submission

1. Form presented inline (from procedure or direct query)
2. Employee fills fields
3. Submit creates submission record
4. Submission embedded for future search
5. Routing: notification to approver/processor
6. Status tracked: pending → approved → completed

### 5.3 Record Retrieval

1. Employee asks: "Show my pending requests" or "POs over $1,000 this quarter"
2. System searches submission embeddings
3. Returns matching records with status

### 5.4 Authoring

1. Author opens editor at `/editor/new`
2. Selects content type (procedure, announcement, form, etc.)
3. For procedures: can link to existing form or create new one
4. System auto-saves and lints with Vale
5. Publish embeds content into knowledge base

### 5.5 Supersession

1. Policy changes require update
2. Author creates new article, marks "supersedes" original
3. Original marked `is_current = false`
4. Queries return current version by default

---

## 6. Technical Architecture

### 6.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Query Interface (primary)                           │    │
│  │  - Ask questions in plain language                   │    │
│  │  - Receive answers + actionable forms                │    │
│  │  - Submit forms, check status                        │    │
│  └──────────────────────┬──────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Authoring Interface                                 │    │
│  │  - Article editor with Vale linting                  │    │
│  │  - Form builder                                      │    │
│  └──────────────────────┬──────────────────────────────┘    │
└─────────────────────────┼────────────────────────────────────┘
                          │ REST API / MCP
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Publishing                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐    │
│  │  REST API   │ │  MCP Server │ │   Embedding         │    │
│  │  /query     │ │  (Claude)   │ │   (OpenAI)          │    │
│  │  /forms     │ │             │ │                     │    │
│  │  /submit    │ └─────────────┘ └─────────────────────┘    │
│  └─────────────┘                                            │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  PostgreSQL + pgvector                               │    │
│  │  articles, training, forms, submissions (all embedded)│    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Database Schema (MVP)

```sql
-- Newsletter issues
CREATE TABLE issues (
    id SERIAL PRIMARY KEY,
    issue_number INTEGER UNIQUE NOT NULL,
    title VARCHAR(255),
    status VARCHAR(20) DEFAULT 'draft',  -- draft, published
    publish_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Articles (reference content)
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    issue_id INTEGER REFERENCES issues(id),

    -- Content
    title VARCHAR(255) NOT NULL,
    slug VARCHAR(100),
    article_type VARCHAR(20) NOT NULL,  -- procedure, story, spotlight, announcement, tutorial
    body TEXT,
    summary TEXT,

    -- Link to actionable form (optional)
    form_id INTEGER REFERENCES forms(id),

    -- Categorization
    topics JSONB DEFAULT '[]',
    division VARCHAR(50),

    -- Supersession
    supersedes_id INTEGER REFERENCES articles(id),
    is_current BOOLEAN DEFAULT TRUE,

    -- Metadata
    author VARCHAR(100),
    effective_date DATE,
    status VARCHAR(20) DEFAULT 'draft',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Forms (actionable templates)
CREATE TABLE forms (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE,
    description TEXT,

    -- Field definitions
    fields JSONB NOT NULL,  -- [{name, type, required, options, default}]

    -- Routing
    route_to VARCHAR(255),  -- email or role

    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Submissions (completed forms = operational records)
CREATE TABLE submissions (
    id SERIAL PRIMARY KEY,
    form_id INTEGER REFERENCES forms(id) NOT NULL,

    -- Submitter
    submitted_by VARCHAR(100) NOT NULL,
    submitted_at TIMESTAMP DEFAULT NOW(),

    -- Data
    field_values JSONB NOT NULL,

    -- Workflow status
    status VARCHAR(20) DEFAULT 'pending',  -- pending, approved, rejected, completed
    processed_by VARCHAR(100),
    processed_at TIMESTAMP,
    notes TEXT,

    -- For embedding/search
    search_text TEXT,  -- denormalized for embedding

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_articles_supersedes ON articles(supersedes_id);
CREATE INDEX idx_articles_current ON articles(is_current) WHERE is_current = TRUE;
CREATE INDEX idx_submissions_status ON submissions(status);
CREATE INDEX idx_submissions_form ON submissions(form_id);
CREATE INDEX idx_submissions_submitter ON submissions(submitted_by);
```

### 6.3 REST API Endpoints

**Reference Content**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/articles` | GET | List articles (filterable) |
| `/api/articles` | POST | Create article |
| `/api/articles/:id` | GET | Get single article |
| `/api/articles/:id` | PUT | Update article |
| `/api/articles/:id/supersede` | POST | Create superseding article |
| `/api/issues` | GET | List issues |
| `/api/issues` | POST | Create issue |
| `/api/issues/:id/publish` | POST | Publish issue |

**Operational Content**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/forms` | GET | List available forms |
| `/api/forms` | POST | Create form template |
| `/api/forms/:id` | GET | Get form with field definitions |
| `/api/submissions` | GET | List submissions (filterable by status, submitter) |
| `/api/submissions` | POST | Submit completed form |
| `/api/submissions/:id` | GET | Get submission details |
| `/api/submissions/:id/status` | PUT | Update submission status |

**Search & Query**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/query` | POST | Semantic search across all content |
| `/api/lint` | POST | Run Vale on content |

### 6.4 MCP Tools

| Tool | Purpose |
|------|---------|
| `query` | Semantic search — returns content AND presents relevant forms |
| `form_submit` | Submit a form, create record |
| `submission_status` | Check status of submissions |
| `article_create` | Create new article |
| `article_draft` | Generate article draft using skill |
| `form_create` | Create new form template |
| `issue_publish` | Generate HTML/PDF output |

---

## 7. Writing Style System

### 7.1 Two-Layer Enforcement

| Layer | Tool | Purpose |
|-------|------|---------|
| Generation | Distillyzer Skill | Guide Claude to write correctly |
| Validation | Vale | Catch violations, block publish |

**Vale is authoritative.** The skill helps Claude write well on first pass, but Vale has final say. Content does not publish until Vale passes.

### 7.2 Vale Configuration

```ini
# .vale.ini
StylesPath = styles
MinAlertLevel = warning

Packages = write-good, proselint, Readability

[*.md]
BasedOnStyles = Vale, Inspire, write-good, proselint, Readability
```

### 7.3 Custom Vale Rules (styles/Inspire/)

**Inspire/NoLists.yml**
```yaml
extends: existence
message: "Do not use bullet points or numbered lists. Write in prose."
level: error
scope: raw
tokens:
  - '^\s*[-*•]\s'
  - '^\s*\d+\.\s'
```

**Inspire/NoHedging.yml**
```yaml
extends: substitution
message: "Avoid hedging. Be direct. Remove '%s'."
level: warning
swap:
  'It is important to note that': ''
  'It should be noted that': ''
  'It is worth mentioning': ''
  'Interestingly enough': ''
  'In order to': 'To'
  'Due to the fact that': 'Because'
  'At this point in time': 'Now'
  'In the event that': 'If'
```

**Inspire/NoMetaCommentary.yml**
```yaml
extends: existence
message: "Remove meta-commentary. Just say it."
level: error
tokens:
  - "Here's a breakdown"
  - "Here are some"
  - "Let's explore"
  - "Let's dive into"
  - "Let's take a look"
  - "There are several"
  - "There are many"
  - "I'd be happy to"
  - "I'll explain"
  - "Allow me to"
```

**Inspire/LeadWithPoint.yml**
```yaml
extends: existence
message: "First sentence may be burying the lead. Start with the outcome or news."
level: suggestion
scope: paragraph
first: true
tokens:
  - '^(In today|In the|When it comes to|As we all know|It goes without saying)'
```

**Inspire/Numbers.yml**
```yaml
extends: existence
message: "Spell out numbers one through nine."
level: warning
tokens:
  - '\b[1-9]\b(?!\s*(percent|%|years old|a\.m\.|p\.m\.|:\d))'
```

**Inspire/CompanyName.yml**
```yaml
extends: substitution
message: "Use 'Inspire Solutions' on first reference, 'Inspire' thereafter."
level: suggestion
swap:
  'Inspire Event Technologies': 'Inspire Solutions'
```

### 7.4 Distillyzer Skill: Inspire Style Guide

```markdown
# Inspire Solutions Writing Style

You are writing for Inspire Record, the internal newsletter of Inspire Solutions. Your writing must be clear, direct, and professional — like journalism, not like an AI.

## ABSOLUTE PROHIBITIONS

Never use:
- Bullet points or numbered lists
- Headers within articles (only the headline exists)
- "Here's..." / "Let's..." / "There are several..."
- "It's important to note..." / "It should be noted..."
- "I'd be happy to..." / "Allow me to..."
- Meta-commentary about what you're about to say
- Concluding summaries that repeat what you just said
- Em dashes for dramatic pauses (use them sparingly for parenthetical information)
- Exclamation points
- Questions as transitions ("So what does this mean?")

## REQUIRED STRUCTURE

**First sentence:** The outcome, the news, the point. What happened or what matters.

**Second sentence:** The key supporting fact. Evidence or context for the first sentence.

**Remaining paragraphs:** Additional context, quotes, details. Each paragraph 1-3 sentences.

**Final paragraph:** Next step, implication, or forward-looking statement. No summary.

## PARAGRAPHS

- 1-3 sentences maximum
- No paragraph longer than 4 lines on screen
- Each paragraph should contain one idea
- Transitions between paragraphs should be implicit, not signposted

## NUMBERS

- Spell out one through nine
- Use numerals for 10 and above
- Always use numerals for: percentages, ages, money, measurements, times

## PUNCTUATION

- No Oxford comma (red, white and blue)
- One space after periods
- Em dashes with spaces — like this — for parenthetical information
- Use quotation marks for direct quotes only

## VOICE

- Active voice required ("The team completed" not "was completed by the team")
- Second person for instructions ("Submit your request" not "Requests should be submitted")
- Third person for news and stories

## COMPANY CONVENTIONS

- "Inspire Solutions" on first reference, "Inspire" thereafter
- Job titles: capitalize before name (Director of Event Technology Mike Chen), lowercase after (Mike Chen, director of event technology)
- Property names: full name on first reference

## EXAMPLE

BAD:
"In today's fast-paced event technology landscape, it's important to understand that there are several key factors that contribute to successful setups. Here's a breakdown of what our team discovered:

- First, preparation is essential
- Second, communication matters
- Third, checklists help

Let me know if you'd like me to elaborate on any of these points!"

GOOD:
"The Dallas team cut setup time by 40 percent last quarter after switching to a checklist-based workflow.

Director Mike Chen tested the approach at three events in October. Crews finished an average of two hours earlier without errors.

'We stopped assuming everyone knew the sequence,' Chen said. 'Writing it down fixed most of our problems.'

The operations team will roll out the checklist to all Southwest properties by March."
```

---

## 8. HTML Template

The newsletter uses a newspaper-style layout with:
- Masthead with publication name, issue number, date, location
- Headline and deck (subheadline)
- Byline
- Two-column body layout
- Section headers (crossheads) for article type
- Pull quotes for key insights
- Prompt boxes for tutorials (monospace, bordered)
- Key takeaway box
- Footer with next issue teaser and submission info

Template file: `templates/inspire-record.html`

---

## 9. MVP Scope

### 9.1 In Scope

| Feature | Description |
|---------|-------------|
| Semantic query | Ask questions, get answers AND actions |
| Forms | Create form templates, link to procedures |
| Submissions | Submit forms, track status, search records |
| Article editor | Browser-based content authoring |
| Training harvest | Ingest video transcripts into knowledge base |
| Supersession | Track what replaces what |
| Unified search | Policies, training, forms, records in one index |

### 9.2 Out of Scope (Future Versions)

| Feature | Description |
|---------|-------------|
| Workflow automation | Auto-routing, approval chains, notifications |
| SSO authentication | Microsoft 365 integration |
| Per-user filtering | Region/property-specific views |
| Meeting transcription | Harvest Teams recordings |
| Email integration | Send/receive via email |
| External system sync | Push submissions to other systems |

---

## 10. Success Criteria

1. **Action over reference:** At least 3 common workflows converted from "read the docs" to "fill out the form"
2. **Semantic findability:** Employees find answers without knowing exact terminology
3. **Record retrieval:** Submission status checkable via query, not email
4. **Single source:** No policy exists in multiple conflicting versions
5. **Training integration:** Video content searchable alongside written policies

---

## 11. Implementation Phases

### Phase 1: Foundation (Current)
- [x] Remove legacy code (demo command)
- [x] Harvest training videos into knowledge base
- [ ] Database schema: forms, submissions tables
- [ ] Form CRUD operations
- [ ] Submission create and status tracking

### Phase 2: Query Interface
- [ ] Semantic query that returns content + presents forms
- [ ] Submission search (my requests, pending POs, etc.)
- [ ] Link forms to procedures

### Phase 3: Content Authoring
- [ ] Article editor with Vale linting
- [ ] Form builder UI
- [ ] Supersession tracking
- [ ] Issue publishing

### Phase 4: Workflow (Future)
- [ ] Auto-routing to approvers
- [ ] Status notifications
- [ ] Division/region structure
- [ ] Microsoft 365 SSO

---

## 12. Open Questions

1. **Hosting:** Where will the system run? (Internal server, Azure, other)
2. **Authentication:** How do employees log in? (Microsoft 365 SSO, other)
3. **Form routing:** Who receives submissions for each form type?
4. **Existing forms:** Which SharePoint/email workflows to migrate first?
5. **Permissions:** Who can create forms vs. who can only submit?

---

## 13. Future: SaaS Model

Publishing can evolve into a multi-tenant operational knowledge platform.

```
Publishing SaaS
├── Tenant: Inspire Solutions
│   ├── Knowledge Base (policies, training, forms, records)
│   ├── Form Templates (cross-rental, PTO, vendor onboarding)
│   ├── Submission Records (searchable operational data)
│   └── Org Structure (divisions, regions, routing rules)
│
├── Tenant: Other Company
│   ├── Knowledge Base
│   ├── Form Templates
│   ├── Submission Records
│   └── Org Structure
│
└── Platform
    ├── Multi-tenant PostgreSQL + pgvector
    ├── Per-tenant form templates and routing
    ├── SSO per tenant
    └── Billing
```

Inspire is customer #1. Build for Inspire's needs, validate, then generalize.

---

## 14. Appendix

### A. File Inventory

| File | Purpose | Status |
|------|---------|--------|
| `src/publishing/forms.py` | Form and submission operations | To create |
| `alembic/versions/*_forms.py` | Database migration for forms/submissions | To create |
| `templates/inspire-record.html` | Newsletter HTML template | To create |
| `styles/Inspire/*.yml` | Vale rule files | To create |
| `.vale.ini` | Vale configuration | To create |

### B. Publishing Components

| Component | Purpose | Modification |
|-----------|---------|--------------|
| `db.py` | Database operations | Add forms, submissions tables |
| `query.py` | Semantic search | Return forms alongside content |
| `rest_server.py` | REST API | Add form/submission endpoints |
| `mcp_server.py` | MCP tools | Add query, form_submit, submission_status |
| `embed.py` | Embedding | Embed submissions for search |

### C. External Dependencies

| Dependency | Purpose | License |
|------------|---------|---------|
| pgvector | Vector similarity search | PostgreSQL |
| OpenAI | Embeddings (text-embedding-3-small) | Commercial |
| Anthropic | Claude for query synthesis | Commercial |
| Vale | Prose linting | MIT |
