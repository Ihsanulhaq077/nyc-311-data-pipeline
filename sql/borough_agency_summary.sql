-- Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-18
SELECT
            borough,
            COUNT(*) AS total_requests,
            ROUND(AVG(resolution_hours), 2) AS avg_resolution_hours,
            COUNT(DISTINCT complaint_type) AS distinct_complaints,
            ROUND(100.0 * SUM(CASE WHEN latitude IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_with_coordinates
        FROM service_requests
        WHERE borough != 'Unknown Borough'
        GROUP BY borough
        ORDER BY total_requests DESC;
