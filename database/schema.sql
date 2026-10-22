-- Football Intelligence Platform - database schema (SQLite)
-- Applied idempotently by database.db.init_db()

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- teams ----
CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    short_name  TEXT    NOT NULL,
    league      TEXT    NOT NULL,
    country     TEXT    NOT NULL,
    stadium     TEXT,
    color       TEXT    NOT NULL DEFAULT '#22c55e'
);

-- -------------------------------------------------------------- players ----
CREATE TABLE IF NOT EXISTS players (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    position        TEXT    NOT NULL,
    nationality     TEXT    NOT NULL,
    age             INTEGER NOT NULL,
    team_id         INTEGER REFERENCES teams(id) ON DELETE SET NULL,
    height_cm       INTEGER,
    preferred_foot  TEXT    NOT NULL DEFAULT 'дясно',
    market_value    INTEGER NOT NULL DEFAULT 0,
    overall         INTEGER NOT NULL DEFAULT 0,
    potential       INTEGER NOT NULL DEFAULT 0,
    contract_years  INTEGER NOT NULL DEFAULT 2,
    -- technical / physical attributes, all on a 0-99 scale
    pace            INTEGER NOT NULL DEFAULT 50,
    shooting        INTEGER NOT NULL DEFAULT 50,
    passing         INTEGER NOT NULL DEFAULT 50,
    dribbling       INTEGER NOT NULL DEFAULT 50,
    defending       INTEGER NOT NULL DEFAULT 50,
    physical        INTEGER NOT NULL DEFAULT 50,
    vision          INTEGER NOT NULL DEFAULT 50,
    positioning     INTEGER NOT NULL DEFAULT 50,
    composure       INTEGER NOT NULL DEFAULT 50,
    teamwork        INTEGER NOT NULL DEFAULT 50,
    stamina         INTEGER NOT NULL DEFAULT 50,
    discipline      INTEGER NOT NULL DEFAULT 50,
    form            INTEGER NOT NULL DEFAULT 50,
    fitness         INTEGER NOT NULL DEFAULT 50,
    injury_risk     INTEGER NOT NULL DEFAULT 20
);

CREATE INDEX IF NOT EXISTS idx_players_position   ON players(position);
CREATE INDEX IF NOT EXISTS idx_players_age        ON players(age);
CREATE INDEX IF NOT EXISTS idx_players_overall    ON players(overall);
CREATE INDEX IF NOT EXISTS idx_players_team       ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_players_nationality ON players(nationality);

-- --------------------------------------------------------- player_stats ----
CREATE TABLE IF NOT EXISTS player_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id       INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    season          TEXT    NOT NULL,
    competition     TEXT    NOT NULL,
    appearances     INTEGER NOT NULL DEFAULT 0,
    minutes         INTEGER NOT NULL DEFAULT 0,
    goals           INTEGER NOT NULL DEFAULT 0,
    assists         INTEGER NOT NULL DEFAULT 0,
    shots           INTEGER NOT NULL DEFAULT 0,
    shots_on_target INTEGER NOT NULL DEFAULT 0,
    key_passes      INTEGER NOT NULL DEFAULT 0,
    tackles         INTEGER NOT NULL DEFAULT 0,
    interceptions   INTEGER NOT NULL DEFAULT 0,
    duels_won_pct   REAL    NOT NULL DEFAULT 0,
    pass_accuracy   REAL    NOT NULL DEFAULT 0,
    dribbles        INTEGER NOT NULL DEFAULT 0,
    clean_sheets    INTEGER NOT NULL DEFAULT 0,
    yellow_cards    INTEGER NOT NULL DEFAULT 0,
    red_cards       INTEGER NOT NULL DEFAULT 0,
    UNIQUE (player_id, season)
);

CREATE INDEX IF NOT EXISTS idx_stats_player ON player_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_stats_season ON player_stats(season);

-- ------------------------------------------------------------ team_needs ----
-- Scouting requests of a club: which position is missing and with what profile.
CREATE TABLE IF NOT EXISTS team_needs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id     INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    position    TEXT    NOT NULL,
    priority    INTEGER NOT NULL DEFAULT 1,
    min_overall INTEGER NOT NULL DEFAULT 0,
    max_age     INTEGER NOT NULL DEFAULT 40,
    max_value   INTEGER NOT NULL DEFAULT 100000000,
    notes       TEXT
);

CREATE INDEX IF NOT EXISTS idx_needs_team ON team_needs(team_id);

-- ------------------------------------------------------------ model_meta ----
-- Metadata of the trained ML model (metrics + coefficients as JSON).
CREATE TABLE IF NOT EXISTS model_meta (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    trained_at TEXT    NOT NULL,
    metrics    TEXT    NOT NULL,
    weights    TEXT    NOT NULL
);
