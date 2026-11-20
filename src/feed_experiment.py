# Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-20
"""Part C1: Social Media Feed Query Performance Experiment with SQLite.

Simulates a high-throughput social network feed query (posts from followed users,
newest first) with and without a composite index on (author_id, created_at DESC).
Captures EXPLAIN QUERY PLAN execution steps and benchmark timings.
"""

from pathlib import Path
import random
import sqlite3
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "outputs" / "reports"
DB_PATH = REPORTS_DIR / "feed_experiment.sqlite"


def run_experiment(num_users: int = 1500, num_posts: int = 80000, num_followed: int = 150):
    """Run feed retrieval query benchmark with and without composite indexes.

    Parameters
    ----------
    num_users : int
        Total synthetic users in the system.
    num_posts : int
        Total synthetic posts across all users.
    num_followed : int
        Number of authors followed by the active target user (User ID 1).
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    print(f"Initializing SQLite Feed Database at: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Schema setup: Posts and Follows graph
        cursor.executescript("""
            DROP TABLE IF EXISTS posts;
            DROP TABLE IF EXISTS follows;

            CREATE TABLE posts (
                post_id INTEGER PRIMARY KEY,
                author_id INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                likes_count INTEGER DEFAULT 0,
                content TEXT
            );

            CREATE TABLE follows (
                follower_id INTEGER NOT NULL,
                followed_id INTEGER NOT NULL,
                PRIMARY KEY (follower_id, followed_id)
            );
        """)

        print(f"Generating {num_posts} synthetic posts and follow relationships...")
        # Populate posts table with random author and incremental timestamps
        base_time = int(time.time()) - (86400 * 30)  # last 30 days
        posts_data = [
            (
                i + 1,
                random.randint(1, num_users),
                base_time + random.randint(0, 86400 * 30),
                random.randint(0, 5000),
                f"Synthetic post content #{i+1}"
            )
            for i in range(num_posts)
        ]
        cursor.executemany("INSERT INTO posts VALUES (?, ?, ?, ?, ?)", posts_data)

        # Target user 1 follows 150 authors
        followed_authors = random.sample(range(2, num_users + 1), num_followed)
        follows_data = [(1, author_id) for author_id in followed_authors]
        cursor.executemany("INSERT INTO follows VALUES (?, ?)", follows_data)
        conn.commit()

        # Target Query: Retrieve top 20 newest posts from users followed by user 1
        query = """
            SELECT p.post_id, p.author_id, p.created_at, p.likes_count, p.content
            FROM posts p
            JOIN follows f ON p.author_id = f.followed_id
            WHERE f.follower_id = 1
            ORDER BY p.created_at DESC
            LIMIT 20;
        """

        # 1. Benchmark WITHOUT Index
        print("\n--- 1. Evaluating Query WITHOUT Index ---")
        cursor.execute("EXPLAIN QUERY PLAN " + query)
        plan_before_rows = cursor.fetchall()
        plan_before_text = "\n".join(str(row) for row in plan_before_rows)

        # Warmup and timing (average over 10 runs)
        times_before = []
        for _ in range(10):
            t0 = time.perf_counter()
            cursor.execute(query).fetchall()
            times_before.append(time.perf_counter() - t0)
        avg_before = sum(times_before) / len(times_before)
        print(f"Query Plan (Before Index):\n{plan_before_text}")
        print(f"Average Execution Time (No Index): {avg_before * 1000:.3f} ms")

        # 2. Add Composite Index
        print("\n--- 2. Building Index: idx_posts_author_created ON posts(author_id, created_at DESC) ---")
        cursor.execute("CREATE INDEX idx_posts_author_created ON posts(author_id, created_at DESC);")
        conn.commit()

        # 3. Benchmark WITH Composite Index
        cursor.execute("EXPLAIN QUERY PLAN " + query)
        plan_after_rows = cursor.fetchall()
        plan_after_text = "\n".join(str(row) for row in plan_after_rows)

        times_after = []
        for _ in range(10):
            t0 = time.perf_counter()
            cursor.execute(query).fetchall()
            times_after.append(time.perf_counter() - t0)
        avg_after = sum(times_after) / len(times_after)
        print(f"Query Plan (After Index):\n{plan_after_text}")
        print(f"Average Execution Time (With Index): {avg_after * 1000:.3f} ms")

        speedup = (avg_before / avg_after) if avg_after > 0 else 1.0
        print(f"\nSpeedup factor with composite index: {speedup:.2f}x faster!")

        # Write artifacts to files
        (REPORTS_DIR / "feed_query_plan_before.txt").write_text(plan_before_text + "\n", encoding="utf-8")
        (REPORTS_DIR / "feed_query_plan_after.txt").write_text(plan_after_text + "\n", encoding="utf-8")
        timing_summary = (
            f"SQLite Feed Query Experiment Timing Summary\n"
            f"-----------------------------------------\n"
            f"Dataset: {num_posts} posts, {num_users} users, 150 followed authors.\n"
            f"Query: Top 20 newest posts from followed authors.\n"
            f"Average Time Without Index: {avg_before * 1000:.4f} ms ({avg_before:.6f} s)\n"
            f"Average Time With Index:    {avg_after * 1000:.4f} ms ({avg_after:.6f} s)\n"
            f"Performance Improvement:    {speedup:.2f}x speedup\n"
        )
        (REPORTS_DIR / "feed_timings.txt").write_text(timing_summary, encoding="utf-8")
        print(f"Saved artifacts to {REPORTS_DIR}")


if __name__ == "__main__":
    run_experiment()
