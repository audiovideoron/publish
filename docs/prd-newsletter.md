# Inspire Record
## Product Requirements Document

**Company:** Inspire Solutions, Dallas, TX  
**Product:** Internal newsletter and knowledge base system  
**Version:** 1.0 MVP  
**Date:** January 2026

---

## 1. Executive Summary

Inspire Record is an internal publication system that serves as the single source of truth for Inspire Solutions. It replaces scattered email notifications, outdated SharePoint documents, and tribal knowledge with a searchable, versioned, professionally written knowledge base.

The newsletter is the authoring interface. Content published through it accumulates into a knowledge base from which other products (Employee Manual, SOPs, Quick Reference Guides) can be derived.

---

## 2. Problem Statement

**Current State:**
- Policy updates arrive via email, then change a month later
- Multiple versions of procedures exist across SharePoint, email threads, and Slack
- No authoritative source when conflicts arise
- Writing quality varies wildly
- Institutional knowledge leaves when employees leave

**Desired State:**
- Single publication of record for all company communications
- Versioned content with clear supersession ("this replaces that")
- Consistent, professional writing style
- Searchable archive
- Reusable content that feeds multiple output formats

---

## 3. Content Types

| Type | Purpose | Example |
|------|---------|---------|
| **Procedure** | SOPs, policies, how-tos | "How to Submit PTO Requests" |
| **Story** | Problem/solution narratives | "How Dallas Cut Setup Time by 40%" |
| **Spotlight** | Division/team achievements | "Q4 Wins from the AV Team" |
| **Announcement** | Policy changes, new tools | "Updated Travel Reimbursement Policy" |
| **Tutorial** | Learning/instruction (Agentic focus) | "Using AI to Draft Client Proposals" |

---

## 4. Information Architecture

### 4.1 Content Hierarchy

```
Knowledge Base
├── Articles (atomic content units)
│   ├── metadata: type, topics, division, author, effective_date
│   ├── versioning: supersedes, superseded_by, is_current
│   └── body: markdown content
│
├── Issues (periodic publications)
│   ├── issue_number, publish_date
│   └── contains 1-N articles
│
└── Derived Products (generated from articles)
    ├── Employee Manual (procedures where is_current=true)
    ├── SOPs (instructional content)
    └── Quick Reference (summaries)
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

### 5.1 Authoring (MVP)

1. Author opens editor at `/editor/new`
2. Selects article type (procedure, story, spotlight, etc.)
3. Writes content in browser-based editor
4. System auto-saves and lints with Vale
5. Author addresses any style violations
6. Author assigns to issue (new or existing)
7. Editor/reviewer approves
8. Publish generates HTML output

### 5.2 Supersession

1. Policy changes require update
2. Author creates new article
3. Marks "supersedes" → selects original article
4. Original article marked `is_current = false`
5. New article shows "Replaces: [original title, Issue #X]"
6. Derived products (manuals) auto-update on next generation

### 5.3 Reading

1. Employee accesses newsletter URL
2. Views current issue or browses archive
3. Searches for specific topics
4. Links to original issue for any article

---

## 6. Technical Architecture

### 6.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Editor (contenteditable HTML)                       │    │
│  │  - Section-based editing                             │    │
│  │  - Auto-save                                         │    │
│  │  - Inline lint feedback                              │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
└─────────────────────────┼────────────────────────────────────┘
                          │ REST API
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Distillyzer (Extended)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐    │
│  │  REST API   │ │  MCP Server │ │   Vale Integration  │    │
│  │  /articles  │ │  (Claude)   │ │   /api/lint         │    │
│  │  /issues    │ │             │ │                     │    │
│  │  /publish   │ └─────────────┘ └─────────────────────┘    │
│  └─────────────┘                                            │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  PostgreSQL + pgvector                               │    │
│  │  articles, issues, people, divisions, regions...     │    │
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

-- Articles (core content unit)
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    issue_id INTEGER REFERENCES issues(id),
    
    -- Content
    title VARCHAR(255) NOT NULL,
    slug VARCHAR(100),
    article_type VARCHAR(20) NOT NULL,  -- procedure, story, spotlight, announcement, tutorial
    body TEXT,
    summary TEXT,
    
    -- Categorization
    topics JSONB DEFAULT '[]',
    division VARCHAR(50),
    
    -- Supersession
    supersedes_id INTEGER REFERENCES articles(id),
    is_current BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    author VARCHAR(100),
    effective_date DATE,
    
    -- Workflow
    status VARCHAR(20) DEFAULT 'draft',  -- draft, review, published
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for supersession queries
CREATE INDEX idx_articles_supersedes ON articles(supersedes_id);
CREATE INDEX idx_articles_current ON articles(is_current) WHERE is_current = TRUE;
```

### 6.3 REST API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/articles` | GET | List articles (filterable) |
| `/api/articles` | POST | Create article |
| `/api/articles/:id` | GET | Get single article |
| `/api/articles/:id` | PUT | Update article |
| `/api/articles/:id/supersede` | POST | Create superseding article |
| `/api/issues` | GET | List issues |
| `/api/issues` | POST | Create issue |
| `/api/issues/:id` | GET | Get issue with articles |
| `/api/issues/:id/publish` | POST | Publish issue (generate HTML) |
| `/api/lint` | POST | Run Vale on content, return issues |
| `/editor/:id` | GET | Serve editor UI for issue |

### 6.4 MCP Tools (Distillyzer Extension)

| Tool | Purpose |
|------|---------|
| `article_create` | Create new article |
| `article_draft` | Generate article draft using skill |
| `article_lint` | Run Vale, return issues |
| `issue_create` | Create new issue |
| `issue_publish` | Generate HTML/PDF output |
| `issue_list` | List all issues |
| `kb_search` | Search knowledge base |

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
| Article editor | Browser-based, contenteditable sections |
| Article types | procedure, story, spotlight, announcement, tutorial |
| Issue management | Create, edit, publish issues |
| Vale linting | Real-time style enforcement |
| Supersession | Track what replaces what |
| Archive | Browse and search past issues |
| HTML output | Generate publishable newsletter |

### 9.2 Out of Scope (Future Versions)

| Feature | Description |
|---------|-------------|
| SSO authentication | Microsoft 365 integration |
| Per-user filtering | Region/property-specific views |
| Meeting transcription | Harvest Teams recordings |
| Automated derivation | Generate manuals from articles |
| PDF output | Print-ready format |
| Email distribution | Send newsletter via email |

---

## 10. Success Criteria

1. **Adoption:** Newsletter replaces at least 3 types of notification emails within 60 days
2. **Quality:** All published content passes Vale with zero errors
3. **Findability:** Users can locate any policy within 30 seconds via search
4. **Currency:** No policy exists in multiple conflicting versions
5. **Authoring time:** New article takes <30 minutes from draft to publish

---

## 11. Implementation Phases

### Phase 1: Foundation
- [ ] Distillyzer writing skill
- [ ] Vale style package
- [ ] Database schema migration
- [ ] Remove legacy code (demo command)

### Phase 2: Editor
- [ ] REST API endpoints
- [ ] Browser-based editor
- [ ] Auto-save and lint integration
- [ ] Issue publishing

### Phase 3: Archive & Search
- [ ] Article archive view
- [ ] Full-text search
- [ ] Supersession display
- [ ] Topic filtering

### Phase 4: Organizational (Future)
- [ ] Division/region structure
- [ ] Meeting transcription pipeline
- [ ] Microsoft 365 SSO
- [ ] Per-user content filtering

---

## 12. Open Questions

1. **Hosting:** Where will the newsletter application run? (Internal server, Azure, other)
2. **Domain:** What URL will employees use? (record.inspiresolutions.com?)
3. **Approval workflow:** Single approver or multi-stage review?
4. **Notification:** How will employees know a new issue is published?
5. **Permissions:** Who can author vs. who can publish?

---

## 13. Future: SaaS Model

Distillyzer can evolve into a multi-tenant knowledge base and publication platform.

```
Distillyzer SaaS
├── Tenant: Inspire Solutions
│   ├── Publications (Inspire Record, Hospitality Digest...)
│   ├── Knowledge Base
│   ├── Style Guide (Inspire-specific Vale rules)
│   └── Org Structure (divisions, regions)
│
├── Tenant: Other Company
│   ├── Publications
│   ├── Knowledge Base
│   ├── Style Guide
│   └── Org Structure
│
└── Platform
    ├── Multi-tenant PostgreSQL
    ├── Per-tenant Vale packages
    ├── SSO per tenant
    └── Billing
```

Inspire is customer #1. Build for Inspire's needs, validate, then generalize.

---

## 14. Appendix

### A. File Inventory

| File | Purpose | Status |
|------|---------|--------|
| `templates/inspire-record.html` | Newsletter HTML template | To create |
| `styles/Inspire/*.yml` | Vale rule files | To create |
| `.vale.ini` | Vale configuration | To create |
| `alembic/versions/*_newsletter.py` | Database migration | To create |
| `src/distillyzer/newsletter.py` | Newsletter module | To create |

### B. Related Distillyzer Components

| Component | Purpose | Modification |
|-----------|---------|--------------|
| `db.py` | Database operations | Extend with article/issue tables |
| `rest_server.py` | REST API | Extend with newsletter endpoints |
| `mcp_server.py` | MCP tools | Add newsletter tools |
| `skills` table | Skill storage | Store writing skill |

### C. External Dependencies

| Dependency | Purpose | License |
|------------|---------|---------|
| Vale | Prose linting | MIT |
| write-good | Style rules | MIT |
| proselint | Style rules | BSD |
| Readability | Complexity metrics | MIT |
