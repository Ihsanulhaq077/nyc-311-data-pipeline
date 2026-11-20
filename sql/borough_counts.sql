-- Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-18
-- Name: Ihsan Ul Haq | Roll No: TRN-2026-000951 | Date: 2026-11-18
-- Description: Aggregate 311 service request volume by NYC borough, sorted in descending order.

SELECT
    borough,
    COUNT(*) AS request_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM service_requests), 2) AS percentage_of_total
FROM service_requests
WHERE borough IS NOT NULL
GROUP BY borough
ORDER BY request_count DESC;
