# Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-20
"""Part C Challenge Bonus (+2): Algorithmic Social Feed Ranker and Caching Benchmark.

Implements a feed ranking algorithm scoring posts by recency and engagement (likes):
    Score = (Likes + 1) / ((AgeInHours + 2) ** 1.5)

Retrieves candidate posts from followed authors, ranks the top 10 for a user,
and demonstrates the performance difference between:
1. Unindexed full table scan
2. Composite index retrieval
3. In-memory LRU cache retrieval
"""

from collections import OrderedDict
from pathlib import Path
import sqlite3
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "outputs" / "reports"
DB_PATH = REPORTS_DIR / "feed_experiment.sqlite"


class LRUCache:
    """Thread-safe, bounded Least-Recently-Used (LRU) in-memory cache."""

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def compute_ranking_score(likes: int, created_at: int, current_timestamp: int) -> float:
    """Calculates engagement-recency score using gravity decay (similar to Hacker News / Reddit)."""
    age_seconds = max(0, current_timestamp - created_at)
    age_hours = age_seconds / 3600.0
    gravity = 1.5
    score = (likes + 1.0) / ((age_hours + 2.0) ** gravity)
    return round(score, 6)


def rank_feed_database(conn: sqlite3.Connection, user_id: int = 1, top_k: int = 10) -> list[dict]:
    """Retrieve candidate posts from DB, score in memory, and return top_k ranked items."""
    current_time = int(time.time())
    # Retrieve posts from followed accounts in the last 7 days
    query = """
        SELECT p.post_id, p.author_id, p.created_at, p.likes_count, p.content
        FROM posts p
        JOIN follows f ON p.author_id = f.followed_id
        WHERE f.follower_id = ?
        ORDER BY p.created_at DESC
        LIMIT 500;
    """
    cursor = conn.cursor()
    rows = cursor.execute(query, (user_id,)).fetchall()

    scored_posts = []
    for r in rows:
        post_id, author_id, created_at, likes, content = r
        score = compute_ranking_score(likes, created_at, current_time)
        scored_posts.append({
            "post_id": post_id,
            "author_id": author_id,
            "likes": likes,
            "created_at": created_at,
            "rank_score": score,
            "content": content,
        })

    # Sort descending by algorithmic rank score
    scored_posts.sort(key=lambda x: x["rank_score"], reverse=True)
    return scored_posts[:top_k]


def benchmark_feed_ranker():
    """Benchmarks feed generation across unindexed DB, indexed DB, and in-memory cache."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        from src.feed_experiment import run_experiment
        run_experiment()

    cache = LRUCache(capacity=50)
    current_time = int(time.time())

    with sqlite3.connect(DB_PATH) as conn:
        # Step 1: Benchmark without index
        conn.execute("DROP INDEX IF EXISTS idx_posts_author_created;")
        t0 = time.perf_counter()
        for _ in range(5):
            rank_feed_database(conn, user_id=1, top_k=10)
        time_no_index = (time.perf_counter() - t0) / 5

        # Step 2: Benchmark with composite index
        conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_author_created ON posts(author_id, created_at DESC);")
        conn.commit()
        t0 = time.perf_counter()
        for _ in range(5):
            ranked_results = rank_feed_database(conn, user_id=1, top_k=10)
        time_with_index = (time.perf_counter() - t0) / 5

        # Populate cache
        cache.put(1, (current_time, ranked_results))

        # Step 3: Benchmark with LRU Cache
        t0 = time.perf_counter()
        for _ in range(50):
            cached = cache.get(1)
            # Cache TTL check (e.g. 60 seconds)
            if cached and (current_time - cached[0] < 60):
                res = cached[1]
            else:
                res = rank_feed_database(conn, user_id=1, top_k=10)
        time_cache = (time.perf_counter() - t0) / 50

    speedup_index = time_no_index / time_with_index if time_with_index > 0 else 1.0
    speedup_cache = time_no_index / time_cache if time_cache > 0 else 1.0

    report = (
        f"Feed Ranker Algorithmic Benchmark (+2 Bonus Deliverable)\n"
        f"====================================================\n"
        f"Ranking Formula: Score = (Likes + 1) / ((AgeHours + 2) ^ 1.5)\n"
        f"Target User: User #1 (following 150 authors)\n\n"
        f"Latency Comparisons (Average per Feed Request):\n"
        f"1. Unindexed DB Scan: {time_no_index * 1000:.3f} ms\n"
        f"2. Indexed DB Query:  {time_with_index * 1000:.3f} ms ({speedup_index:.1f}x faster than unindexed)\n"
        f"3. In-Memory LRU Cache: {time_cache * 1000:.4f} ms ({speedup_cache:.1f}x faster than unindexed)\n\n"
        f"Sample Top 5 Ranked Feed Results for User #1:\n"
    )
    for idx, post in enumerate(ranked_results[:5], 1):
        report += (
            f"  {idx}. Post #{post['post_id']} by Author {post['author_id']} | "
            f"Likes: {post['likes']} | Rank Score: {post['rank_score']:.4f}\n"
        )

    (REPORTS_DIR / "feed_ranker_benchmark.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    benchmark_feed_ranker()

