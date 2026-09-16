# BIRD ten-question verifier pilot: complete model judgments

## Question 0, candidate 1

What is the highest eligible free rate for K-12 students in the schools in Alameda County?

Evidence: Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

Benchmark execution label: True

```sql
SELECT MAX("Percent (%) Eligible Free (K-12)") AS HighestEligibleFreeRate FROM frpm INNER JOIN schools ON frpm.CDSCode = schools.CDSCode WHERE schools.County = 'Alameda'
```

- **q0:output_semantics: S** — The SQL query computes the maximum value of `Percent (%) Eligible Free (K-12)` for schools in Alameda County, which directly matches the question's request for the highest eligible free rate. The aggregate function is correct, and the target column is properly selected.
- **q0:constraint_semantics: S** — The SQL query correctly filters schools in Alameda County and computes the maximum eligible free rate using the correct column. The WHERE clause properly restricts the population to Alameda County, which is required by the question.
- **q0:ranking_semantics: S** — The question asks for the highest eligible free rate, which is an extremum (MAX) query. The SQL uses MAX on the relevant column and filters by county, satisfying the ranking_semantics rubric.
- **q0:relation_semantics: S** — The query joins the frpm and schools tables on CDSCode, which is a key relationship. It then filters by county and computes the maximum eligible free rate, which requires understanding the relationship between schools and their corresponding frpm data.

## Question 0, candidate 2

What is the highest eligible free rate for K-12 students in the schools in Alameda County?

Evidence: Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

Benchmark execution label: True

```sql
SELECT MAX(T1."Percent (%) Eligible Free (K-12)") AS highest_eligible_free_rate FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Alameda'
```

- **q0:output_semantics: S** — The SQL query computes the maximum value of `Percent (%) Eligible Free (K-12)` for schools in Alameda County, which directly matches the question's request for the highest eligible free rate. The aggregation and projection are correct according to the schema and the question's requirements.
- **q0:constraint_semantics: S** — The SQL correctly filters schools in Alameda County and computes the maximum eligible free rate from the `Percent (%) Eligible Free (K-12)` column, which aligns with the question's requirement.
- **q0:ranking_semantics: S** — The question asks for the highest eligible free rate, which is an extremum (MAX) query. The SQL uses MAX on the relevant column and filters by county, satisfying the ranking_semantics rubric.
- **q0:relation_semantics: S** — The query joins the frpm and schools tables on CDSCode, which is a key relationship. It then filters by County and computes the maximum value of the pre-calculated 'Percent (%) Eligible Free (K-12)' column, which aligns with the question's requirement to find the highest eligible free rate in Alameda County.

## Question 0, candidate 3

What is the highest eligible free rate for K-12 students in the schools in Alameda County?

Evidence: Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

Benchmark execution label: True

```sql
SELECT MAX("Percent (%) Eligible Free (K-12)") FROM frpm WHERE "County Name" = 'Alameda'
```

- **q0:output_semantics: S** — The SQL query correctly computes the maximum value of the precomputed column `Percent (%) Eligible Free (K-12)`, which aligns with the question's request for the highest eligible free rate. The column is already a ratio (Free Meal Count / Enrollment), so no further computation is needed.
- **q0:constraint_semantics: S** — The SQL query correctly filters for schools in Alameda County and computes the maximum eligible free rate as required by the question. The WHERE clause correctly applies the county filter, and the MAX function is appropriately used on the correct column.
- **q0:ranking_semantics: S** — The question asks for the highest eligible free rate, which is an extremum (MAX) request. The SQL uses MAX on the relevant column and filters by 'Alameda County', satisfying the ranking_semantics rubric.
- **q0:relation_semantics: U** — The query calculates the maximum value of 'Percent (%) Eligible Free (K-12)' for schools in Alameda County, which requires grouping by the county to ensure the correct aggregation. However, the query does not explicitly group by 'County Name', which is used in the WHERE clause. This could lead to incorrect results if multiple counties have the same name or if the aggregation is not properly scoped.

## Question 0, candidate 4

What is the highest eligible free rate for K-12 students in the schools in Alameda County?

Evidence: Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

Benchmark execution label: False

```sql
SELECT MAX((T1."Free Meal Count (K-12)" / T1."Enrollment (K-12)" * 100)) AS max_eligible_free_rate FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Alameda'
```

- **q0:output_semantics: S** — The SQL computes the maximum of the eligible free rate (as a percentage) for schools in Alameda County, which matches the question's request. The formula used is correct as per the provided evidence, and the output structure aligns with the required semantics.
- **q0:constraint_semantics: S** — The SQL correctly filters schools in Alameda County using the WHERE clause and computes the eligible free rate as required by the question. The MAX function is appropriately applied to find the highest rate.
- **q0:ranking_semantics: S** — The question asks for the highest eligible free rate, which is an extremum (MAX) calculation. The SQL uses MAX on the computed rate, which aligns with the requirement. The query correctly filters schools in Alameda County and calculates the required rate.
- **q0:relation_semantics: S** — The query joins the frpm and schools tables on CDSCode, which is a key relationship. It then filters by county and calculates the maximum eligible free rate, which requires grouping or aggregating across schools in Alameda County. This satisfies the relationship requirement.

## Question 1, candidate 3

Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.

Evidence: Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

Benchmark execution label: False

```sql
SELECT "Percent (%) Eligible Free (Ages 5-17)" FROM frpm WHERE "School Type" = 'Continuation' ORDER BY "Percent (%) Eligible Free (Ages 5-17)" ASC LIMIT 3
```

- **q0:output_semantics: S** — The SQL query correctly selects the 'Percent (%) Eligible Free (Ages 5-17)' column from the frpm table where the 'School Type' is 'Continuation', and orders the results by this percentage in ascending order to get the lowest three values. This matches the question's request for the lowest three eligible free rates for students aged 5-17 in continuation schools.
- **q0:constraint_semantics: S** — The SQL query correctly filters for 'School Type' = 'Continuation' and orders the results by the 'Percent (%) Eligible Free (Ages 5-17)' in ascending order, which aligns with the requirement to list the lowest three eligible free rates. The WHERE clause correctly applies the population restriction for continuation schools.
- **q0:ranking_semantics: S** — The question asks for the lowest three eligible free rates for students aged 5-17 in continuation schools, which requires ordering by the relevant percentage and limiting to the top 3 results. The SQL query correctly uses ORDER BY with ASC and LIMIT 3 to achieve this.
- **q0:relation_semantics: NA** — The query retrieves the percentage of eligible free meals for continuation schools but does not involve combining entities, grouping by an entity, or reshaping. It is a single-table scalar selection without a relationship requirement.

## Question 2, candidate 0

Please list the zip code of all the charter schools in Fresno County Office of Education.

Evidence: Charter schools refers to `Charter School (Y/N)` = 1 in the table fprm

Benchmark execution label: False

```sql
SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1."Charter School (Y/N)" = 1 AND T1."County Name" = 'Fresno County Office of Education'
```

- **q0:output_semantics: S** — The SQL query selects the Zip code from the schools table where the Charter School (Y/N) is 1 and the County Name is 'Fresno County Office of Education'. This correctly addresses the question by retrieving the zip codes of charter schools in the specified county.
- **q0:constraint_semantics: S** — The SQL query correctly filters for charter schools (Charter School (Y/N) = 1) and specifies the county as 'Fresno County Office of Education', which aligns with the question's requirements.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks for a list of zip codes of charter schools in a specific county. The SQL query does not include any ranking or ordering semantics, and the task does not require such features.
- **q0:relation_semantics: S** — The query joins the frpm and schools tables on CDSCode, which is a foreign key relationship. It filters for charter schools (Charter School (Y/N) = 1) and Fresno County Office of Education (County Name). This involves combining entities through a relationship, satisfying the rubric's requirement.

## Question 2, candidate 3

Please list the zip code of all the charter schools in Fresno County Office of Education.

Evidence: Charter schools refers to `Charter School (Y/N)` = 1 in the table fprm

Benchmark execution label: False

```sql
SELECT Zip FROM schools WHERE EdOpsCode = 'Fresno County Office of Education' AND Charter = 1
```

- **q0:output_semantics: S** — The SQL query selects the Zip code from the schools table where EdOpsCode is 'Fresno County Office of Education' and Charter is 1. The question asks for the zip code of all charter schools in Fresno County Office of Education. The query correctly identifies the required entities (zip code) and applies the correct filter (Charter = 1). The output semantics match the question's requirements.
- **q0:constraint_semantics: S** — The SQL query correctly filters for schools where `EdOpsCode` is 'Fresno County Office of Education' and `Charter` is 1, which aligns with the requirement to list zip codes of charter schools in Fresno County Office of Education. The constraint is satisfied.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. The SQL query retrieves zip codes of charter schools in Fresno County Office of Education without any ranking or ordering requirements.
- **q0:relation_semantics: S** — The task requires selecting zip codes of charter schools in a specific county. The query filters schools by EdOpsCode and Charter, which involves a relationship between the schools table and the county information. The query does not explicitly join with the frpm table, but it correctly identifies the relevant entity (schools) and applies the necessary filter (Charter = 1) to meet the task's requirements.

## Question 2, candidate 4

Please list the zip code of all the charter schools in Fresno County Office of Education.

Evidence: Charter schools refers to `Charter School (Y/N)` = 1 in the table fprm

Benchmark execution label: False

```sql
SELECT DISTINCT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1."Charter School (Y/N)" = 1 AND T2.County = 'Fresno County Office of Education'
```

- **q0:output_semantics: S** — The SQL query selects the Zip code from the schools table where the Charter School (Y/N) is 1 and the County is 'Fresno County Office of Education'. The question asks for the zip code of all charter schools in Fresno County Office of Education, which matches the output semantics.
- **q0:constraint_semantics: S** — The SQL query correctly filters for charter schools (Charter School (Y/N) = 1) and schools in Fresno County Office of Education (County = 'Fresno County Office of Education'), which aligns with the question's requirements.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks for a list of zip codes of charter schools in a specific county. The SQL query does not include any ranking or ordering semantics.
- **q0:relation_semantics: S** — The query joins the frpm and schools tables on CDSCode, filters for charter schools (Charter School (Y/N) = 1) and Fresno County Office of Education, and selects the Zip code. This involves combining entities (frpm and schools) and grouping by the County, which satisfies the relation_semantics rubric.

## Question 3, candidate 0

What is the unabbreviated mailing address of the school with the highest FRPM count for K-12 students?

Evidence: (none)

Benchmark execution label: False

```sql
SELECT s.MailStreet || ' ' || s.MailStrAbr || ', ' || s.MailCity || ', ' || s.MailZip || ', ' || s.MailState AS mailing_address FROM schools AS s INNER JOIN frpm AS f ON s.CDSCode = f.CDSCode ORDER BY f."FRPM Count (K-12)" DESC LIMIT 1
```

- **q0:output_semantics: S** — The SQL query correctly selects the unabbreviated mailing address by concatenating the relevant fields from the schools table. The projection matches the requested entity, and there are no issues with the aggregate functions or output structure.
- **q0:constraint_semantics: S** — The SQL query correctly filters for the school with the highest FRPM Count (K-12) using the ORDER BY clause and LIMIT 1, which aligns with the question's requirement to find the school with the highest FRPM count. The constraint on the FRPM count is properly applied in the ORDER BY clause.
- **q0:ranking_semantics: S** — The question asks for the school with the highest FRPM count, which requires identifying the top 1 school based on that count. The SQL includes an ORDER BY clause with DESC and a LIMIT 1, which correctly implements this ranking requirement.
- **q0:relation_semantics: S** — The query joins the schools and frpm tables on CDSCode, groups the data by school (implied by the ORDER BY and LIMIT 1), and retrieves the mailing address of the school with the highest FRPM count. This satisfies the requirement of combining entities and grouping by an entity.

## Question 3, candidate 1

What is the unabbreviated mailing address of the school with the highest FRPM count for K-12 students?

Evidence: (none)

Benchmark execution label: False

```sql
SELECT T2.MailStreet, T2.MailStrAbr, T2.MailCity, T2.MailZip, T2.MailState FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1."FRPM Count (K-12)" DESC LIMIT 1
```

- **q0:output_semantics: S** — The SQL query selects the mailing address fields (MailStreet, MailStrAbr, MailCity, MailZip, MailState) from the schools table, joined with the frpm table on CDSCode. The output is ordered by the highest FRPM Count (K-12) and limited to one row, which aligns with the question's request for the unabbreviated mailing address of the school with the highest FRPM count. The selected fields correctly represent the mailing address components.
- **q0:constraint_semantics: S** — The SQL query correctly joins the frpm and schools tables on CDSCode and orders by the FRPM Count (K-12) in descending order to find the school with the highest FRPM count. It then selects the mailing address fields from the schools table. The constraint on the FRPM count is properly applied in the ORDER BY clause.
- **q0:ranking_semantics: S** — The question asks for the school with the highest FRPM count, which requires an extremum (maximum) and ordering. The SQL uses ORDER BY with DESC and LIMIT 1 to retrieve the top result, satisfying the ranking semantics requirement.
- **q0:relation_semantics: S** — The query joins the frpm and schools tables on CDSCode, which is a foreign key relationship. It then orders by the FRPM Count (K-12) to find the school with the highest count and selects the mailing address fields from the schools table. This satisfies the requirement of combining entities and grouping by an entity (implied by ordering and limiting to one row).

## Question 3, candidate 2

What is the unabbreviated mailing address of the school with the highest FRPM count for K-12 students?

Evidence: (none)

Benchmark execution label: False

```sql
SELECT T1.MailStreet, T1.MailCity, T1.MailState, T1.MailZip FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T2."FRPM Count (K-12)" DESC LIMIT 1
```

- **q0:output_semantics: S** — The SQL query selects the mailing address fields (MailStreet, MailCity, MailState, MailZip) from the schools table, joined with the frpm table. The output is ordered by the highest FRPM Count (K-12) and limited to one row, which aligns with the question's request for the unabbreviated mailing address of the school with the highest FRPM count. The projection matches the required entities.
- **q0:constraint_semantics: S** — The SQL query correctly joins the schools and frpm tables on CDSCode and orders by the FRPM Count (K-12) in descending order to find the school with the highest FRPM count. The required filtering and ordering are present to meet the question's requirement.
- **q0:ranking_semantics: S** — The question asks for the unabbreviated mailing address of the school with the highest FRPM count, which requires identifying the top school based on FRPM Count (K-12). The SQL query uses ORDER BY with DESC and LIMIT 1 to achieve this, satisfying the ranking semantics required.
- **q0:relation_semantics: S** — The query joins the schools and frpm tables on CDSCode, which is a foreign key relationship, and groups the results by the FRPM Count (K-12) to find the school with the highest count. This satisfies the requirement of combining entities and grouping by an entity.

## Question 3, candidate 3

What is the unabbreviated mailing address of the school with the highest FRPM count for K-12 students?

Evidence: (none)

Benchmark execution label: False

```sql
SELECT MailStreet, MailCity, MailState, MailZip FROM schools WHERE CDSCode = (SELECT CDSCode FROM frpm ORDER BY "FRPM Count (K-12)" DESC LIMIT 1)
```

- **q0:output_semantics: S** — The SQL query selects the mailing address fields (MailStreet, MailCity, MailState, MailZip) from the schools table for the school with the highest FRPM Count (K-12). This directly answers the question by retrieving the unabbreviated mailing address of the school with the highest FRPM count.
- **q0:constraint_semantics: S** — The SQL query correctly filters the schools table to retrieve the unabbreviated mailing address of the school with the highest FRPM count by using a subquery to find the CDSCode with the maximum FRPM Count (K-12).
- **q0:ranking_semantics: S** — The question asks for the school with the highest FRPM count, which is an extremum (maximum) request. The SQL uses an ORDER BY with DESC and LIMIT 1 to find the top school, satisfying the ranking semantics requirement.
- **q0:relation_semantics: S** — The query selects the mailing address from the schools table based on the CDSCode from the frpm table, which is ordered by the highest FRPM Count. This requires a relationship between the schools and frpm tables through the CDSCode, which is correctly used as a foreign key. The query satisfies the requirement of grouping or relating entities.
- **q1:output_semantics: S** — The SQL query correctly selects the CDSCode from the frpm table, ordered by 'FRPM Count (K-12)' in descending order and limited to 1, which is the required step to find the school with the highest FRPM count. The projection of MailStreet, MailCity, MailState, and MailZip from the schools table is appropriate for retrieving the unabbreviated mailing address.
- **q1:constraint_semantics: S** — The subquery correctly orders the frpm table by 'FRPM Count (K-12)' in descending order and limits to the top result, which is the required population for finding the school with the highest FRPM count. The WHERE clause in the main query correctly uses this CDSCode to filter the schools table.
- **q1:ranking_semantics: S** — The question asks for the school with the highest FRPM count, which requires identifying the top 1 school based on that count. The SQL uses ORDER BY with DESC and LIMIT 1 to achieve this, satisfying the ranking semantics required.
- **q1:relation_semantics: S** — The query requires selecting the CDSCode from the frpm table, ordering by 'FRPM Count (K-12)' to find the highest value, which involves a relationship between the frpm and schools tables through the CDSCode. This satisfies the requirement of grouping or combining entities.

## Question 3, candidate 4

What is the unabbreviated mailing address of the school with the highest FRPM count for K-12 students?

Evidence: (none)

Benchmark execution label: False

```sql
SELECT s.MailStreet || ', ' || s.MailStrAbr || ', ' || s.MailCity || ', ' || s.MailZip || ', ' || s.MailState AS FullAddress FROM frpm AS f JOIN schools AS s ON f.CDSCode = s.CDSCode ORDER BY f."FRPM Count (K-12)" DESC NULLS FIRST LIMIT 1
```

- **q0:output_semantics: S** — The SQL query selects the unabbreviated mailing address by concatenating MailStreet, MailStrAbr, MailCity, MailZip, and MailState fields from the schools table. This matches the question's request for the unabbreviated mailing address of the school with the highest FRPM count.
- **q0:constraint_semantics: S** — The SQL query correctly filters for the school with the highest FRPM Count (K-12) using the ORDER BY clause with DESC and LIMIT 1, which satisfies the constraint of identifying the school with the highest FRPM count.
- **q0:ranking_semantics: S** — The question asks for the unabbreviated mailing address of the school with the highest FRPM count, which requires identifying the top school based on FRPM Count (K-12). The SQL includes an ORDER BY clause with DESC and LIMIT 1, which correctly identifies the top school. The ranking semantics are satisfied.
- **q0:relation_semantics: S** — The query joins the frpm and schools tables on CDSCode, which is a foreign key relationship, and groups the data by the FRPM Count (K-12) to find the school with the highest count. This satisfies the requirement of combining entities and grouping by an entity.

## Question 4, candidate 0

Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.

Evidence: Charter schools refers to `Charter School (Y/N)` = 1 in the frpm

Benchmark execution label: False

```sql
SELECT s.Phone FROM schools AS s INNER JOIN frpm AS f ON s.CDSCode = f.CDSCode WHERE f."Charter School (Y/N)" = 1 AND s.OpenDate > '2000-01-01'
```

- **q0:output_semantics: S** — The SQL query correctly selects the phone numbers from the schools table where the school is a charter school (Charter School (Y/N) = 1) and was opened after 2000/1/1. The projection matches the requested entities (phone numbers) and the filters align with the question's requirements.
- **q0:constraint_semantics: S** — The SQL correctly filters for direct charter-funded schools (Charter School (Y/N) = 1) and ensures they were opened after 2000/1/1 by checking s.OpenDate > '2000-01-01'.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks for phone numbers of direct charter-funded schools opened after 2000/1/1. The SQL query does not include any ranking or ordering semantics.
- **q0:relation_semantics: S** — The query joins the schools and frpm tables on CDSCode, filters for charter schools (Charter School (Y/N) = 1) and schools opened after 2000/1/1, and selects phone numbers. This involves combining entities (schools and frpm) through a relationship (CDSCode) and grouping by the school entity (implied by the join and filtering).

## Question 4, candidate 1

Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.

Evidence: Charter schools refers to `Charter School (Y/N)` = 1 in the frpm

Benchmark execution label: False

```sql
SELECT T1.Phone FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2."Charter School (Y/N)" = 1 AND T1.OpenDate > '2000-01-01'
```

- **q0:output_semantics: S** — The SQL query correctly selects the phone numbers from the schools table where the school is a charter school (Charter School (Y/N) = 1) and was opened after 2000/1/1. The projection (Phone) and filtering conditions align with the question's requirements.
- **q0:constraint_semantics: S** — The SQL correctly filters for direct charter-funded schools (Charter School (Y/N) = 1) and ensures they were opened after 2000/1/1 (OpenDate > '2000-01-01'). The Boolean logic and WHERE clause scope are preserved as required.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks for phone numbers of direct charter-funded schools opened after 2000/1/1. The SQL query does not include any ranking or ordering semantics.
- **q0:relation_semantics: S** — The query joins the schools and frpm tables on CDSCode and filters for charter schools opened after 2000/1/1. This involves combining entities (schools and frpm) and grouping by the school entity (via the join), which satisfies the relation_semantics rubric.

## Question 4, candidate 2

Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.

Evidence: Charter schools refers to `Charter School (Y/N)` = 1 in the frpm

Benchmark execution label: False

```sql
SELECT T1.Phone FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.Charter = 1 AND T1.OpenDate > '2000-01-01'
```

- **q0:output_semantics: S** — The SQL query selects the phone numbers of schools where Charter = 1 and OpenDate > '2000-01-01'. The question asks for the phone numbers of direct charter-funded schools opened after 2000/1/1. The query correctly identifies the required entities (phone numbers) and filters based on the correct conditions (Charter = 1 and OpenDate > '2000-01-01').
- **q0:constraint_semantics: S** — The SQL correctly filters for direct charter-funded schools (Charter = 1) and ensures they were opened after 2000/1/1 (OpenDate > '2000-01-01'), which matches the question's requirements.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks for phone numbers of direct charter-funded schools opened after 2000/1/1. The SQL query does not include any ranking or ordering semantics.
- **q0:relation_semantics: S** — The query joins the schools and frpm tables on CDSCode, which is a foreign key relationship. It filters for schools that are charter schools (Charter = 1) and opened after 2000/1/1. This involves combining entities through a relationship and grouping by the school entity (implied by the filter on Charter and OpenDate).

## Question 4, candidate 3

Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.

Evidence: Charter schools refers to `Charter School (Y/N)` = 1 in the frpm

Benchmark execution label: False

```sql
SELECT Phone FROM schools WHERE Charter = 1 AND OpenDate > '2000-01-01'
```

- **q0:output_semantics: S** — The SQL query correctly selects the 'Phone' column from the 'schools' table where 'Charter' is 1 (indicating a charter school) and 'OpenDate' is after '2000-01-01'. This matches the question's request for the phone numbers of direct charter-funded schools opened after 2000/1/1. The output semantics are satisfied as the projection and filtering align with the question's requirements.
- **q0:constraint_semantics: S** — The SQL query correctly filters for direct charter-funded schools (Charter = 1) and opens after 2000/1/1 (OpenDate > '2000-01-01'), which aligns with the question's requirements.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks for phone numbers of direct charter-funded schools opened after 2000/1/1. The SQL query correctly filters the data without any ranking or ordering requirements.
- **q0:relation_semantics: S** — The task requires filtering schools that are charter schools (Charter = 1) and opened after 2000/1/1. The query correctly applies these filters on the schools table, which contains the necessary information (Charter and OpenDate). There is no need for joining other tables or grouping, as the question only asks for phone numbers of direct charter-funded schools meeting the criteria.

## Question 4, candidate 4

Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.

Evidence: Charter schools refers to `Charter School (Y/N)` = 1 in the frpm

Benchmark execution label: False

```sql
SELECT s.Phone FROM schools AS s INNER JOIN frpm AS f ON s.CDSCode = f.CDSCode WHERE s.OpenDate > '2000-01-01' AND f."Charter School (Y/N)" = 1
```

- **q0:output_semantics: S** — The SQL query correctly selects the phone numbers from the schools table where the school is a charter school (Charter School (Y/N) = 1) and was opened after 2000/1/1. The projection matches the requested entities (phone numbers) and the filters align with the question's criteria.
- **q0:constraint_semantics: S** — The SQL correctly filters for schools opened after 2000/1/1 and charter schools (Charter School (Y/N) = 1). The constraints are properly applied in the WHERE clause and align with the question's requirements.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks for phone numbers of direct charter-funded schools opened after 2000/1/1. The SQL query does not include any ranking or ordering semantics.
- **q0:relation_semantics: S** — The query joins the schools and frpm tables on CDSCode, which is a foreign key relationship, and filters for schools opened after 2000/1/1 and charter schools. This satisfies the requirement of combining entities and grouping by an entity (schools) based on their relationship.

## Question 5, candidate 0

How many schools with an average score in Math under 400 in the SAT test are exclusively virtual?

Evidence: Exclusively virtual refers to Virtual = 'F'

Benchmark execution label: True

```sql
SELECT COUNT(*) AS TotalVirtualSchools FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.AvgScrMath < 400 AND T1.Virtual = 'F'
```

- **q0:output_semantics: S** — The SQL query counts the number of schools that are exclusively virtual (Virtual = 'F') and have an average Math score under 400. This matches the question's request for the count of such schools.
- **q0:constraint_semantics: S** — The SQL query correctly applies the required filters: T2.AvgScrMath < 400 (average Math score under 400) and T1.Virtual = 'F' (exclusively virtual schools). These conditions align with the question's requirements.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It asks for a count of schools meeting specific criteria, which does not involve ranking semantics.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, which is a key relationship. It filters for schools with Virtual = 'F' and average math scores under 400, indicating a relationship between entities and grouping by school type.

## Question 5, candidate 1

How many schools with an average score in Math under 400 in the SAT test are exclusively virtual?

Evidence: Exclusively virtual refers to Virtual = 'F'

Benchmark execution label: True

```sql
SELECT COUNT(*) FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.AvgScrMath < 400 AND T1.Virtual = 'F'
```

- **q0:output_semantics: S** — The SQL query counts the number of schools that have an average Math score under 400 and are exclusively virtual (Virtual = 'F'). This directly matches the question's request for the count of such schools.
- **q0:constraint_semantics: S** — The SQL query correctly applies the required filters: it checks for schools with an average Math score under 400 (T2.AvgScrMath < 400) and exclusively virtual schools (T1.Virtual = 'F'). These conditions align with the question's requirements.
- **q0:ranking_semantics: NA** — The question asks for the count of schools with an average Math score under 400 and being exclusively virtual. The SQL query uses COUNT(*) without any ordering, ranking, or limit clause, which does not satisfy the ranking semantics required by the question. The query does not involve any ranking, ordering, or extremum operations.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, which is a key relationship. It filters for schools with Virtual = 'F' and average math scores under 400, which involves combining entities and grouping by the school entity (implied by the join and filtering).

## Question 5, candidate 2

How many schools with an average score in Math under 400 in the SAT test are exclusively virtual?

Evidence: Exclusively virtual refers to Virtual = 'F'

Benchmark execution label: True

```sql
SELECT COUNT(T1.CDSCode) FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.AvgScrMath < 400 AND T1.Virtual = 'F'
```

- **q0:output_semantics: S** — The SQL query correctly counts the number of schools with an average Math score under 400 and where Virtual is 'F', which matches the question's requirements.
- **q0:constraint_semantics: S** — The SQL query correctly applies the required filters: T2.AvgScrMath < 400 (average Math score under 400) and T1.Virtual = 'F' (exclusively virtual schools). These conditions align with the question's requirements.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It asks for a count of schools meeting specific criteria, which does not involve ranking semantics.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, which is a key relationship. It filters based on the average math score and virtual status, indicating a relationship requirement between entities.

## Question 5, candidate 3

How many schools with an average score in Math under 400 in the SAT test are exclusively virtual?

Evidence: Exclusively virtual refers to Virtual = 'F'

Benchmark execution label: True

```sql
SELECT COUNT(DISTINCT T1.CDSCode) FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.AvgScrMath < 400 AND T1.Virtual = 'F'
```

- **q0:output_semantics: S** — The SQL query correctly counts distinct schools (CDSCode) that have an average Math score under 400 and are exclusively virtual (Virtual = 'F'). The aggregate function COUNT(DISTINCT) is appropriate for counting unique schools, and the conditions match the question's requirements.
- **q0:constraint_semantics: S** — The SQL query correctly applies the required filters: T2.AvgScrMath < 400 (average Math score under 400) and T1.Virtual = 'F' (exclusively virtual schools). These conditions align with the question's requirements.
- **q0:ranking_semantics: NA** — The question asks for the count of schools with an average Math score under 400 that are exclusively virtual. The SQL query uses COUNT(DISTINCT) without any ordering or ranking, and the rubric specifies that a sum/count/average alone is not considered ranking. Since there is no explicit ranking or ordering requirement, the label is NA.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, which is a key relationship. It filters based on the average math score and virtual status, and counts distinct schools. This satisfies the requirement of combining entities and grouping by an entity (Virtual = 'F').

## Question 6, candidate 0

Among the schools with the SAT test takers of over 500, please list the schools that are magnet schools or offer a magnet program.

Evidence: Magnet schools or offer a magnet program means that Magnet = 1

Benchmark execution label: True

```sql
SELECT DISTINCT s.School FROM schools AS s INNER JOIN satscores AS ss ON s.CDSCode = ss.cds WHERE ss.NumTstTakr > 500 AND s.Magnet = 1
```

- **q0:output_semantics: S** — The SQL query selects schools with SAT test takers over 500 and Magnet = 1, which matches the question's requirement. The projection is correct as it lists the schools, and the filter on Magnet = 1 correctly identifies magnet schools.
- **q0:constraint_semantics: S** — The SQL correctly filters schools with SAT test takers over 500 (ss.NumTstTakr > 500) and magnet schools (s.Magnet = 1), which aligns with the question's requirement. The WHERE clause correctly applies the constraints.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks to list schools that meet certain criteria (SAT test takers over 500 and magnet schools). The SQL query does not include any ranking or ordering semantics.
- **q0:relation_semantics: S** — The query requires joining the schools and satscores tables based on CDSCode to filter schools with SAT test takers over 500 and Magnet = 1. This involves combining entities and grouping by the school entity (via the join and filter conditions).

## Question 6, candidate 2

Among the schools with the SAT test takers of over 500, please list the schools that are magnet schools or offer a magnet program.

Evidence: Magnet schools or offer a magnet program means that Magnet = 1

Benchmark execution label: True

```sql
SELECT T1.School FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.NumTstTakr > 500 AND T1.Magnet = 1
```

- **q0:output_semantics: S** — The SQL query selects the 'School' column from the 'schools' table where the number of SAT test takers is over 500 and the school is a magnet school (Magnet = 1). This matches the question's requirement to list schools with over 500 SAT test takers that are magnet schools or offer a magnet program. The projection and filtering are correct under the given schema.
- **q0:constraint_semantics: S** — The SQL query correctly filters schools with more than 500 SAT test takers (NumTstTakr > 500) and magnet schools (Magnet = 1), which aligns with the question's requirement to list schools with magnet programs.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks to list schools that meet certain criteria (SAT test takers over 500 and being magnet schools). The SQL query does not include any ranking or ordering semantics.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, filters for schools with over 500 SAT test takers and Magnet = 1, which satisfies the requirement of combining entities and grouping by the school entity (via the join and filter on Magnet).

## Question 6, candidate 3

Among the schools with the SAT test takers of over 500, please list the schools that are magnet schools or offer a magnet program.

Evidence: Magnet schools or offer a magnet program means that Magnet = 1

Benchmark execution label: False

```sql
SELECT School, Magnet FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.NumTstTakr > 500 AND T1.Magnet = 1
```

- **q0:output_semantics: S** — The SQL query selects 'School' and 'Magnet' from the schools table where the number of SAT test takers is over 500 and Magnet is 1. The question asks for schools with over 500 SAT test takers that are magnet schools or offer a magnet program. The query correctly filters for Magnet = 1, which aligns with the requirement. However, the query also includes the 'School' column, which is not necessary for the output as the question only asks to list the schools (which would be implied by the filter). The inclusion of 'Magnet' is redundant since it's already used in the filter. The output semantics are satisfied as the query correctly identifies the required schools.
- **q0:constraint_semantics: S** — The SQL query correctly filters schools with over 500 SAT test takers (NumTstTakr > 500) and selects schools that are magnet schools (Magnet = 1), which aligns with the question's requirement.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks to list schools with specific conditions (SAT test takers over 500 and Magnet = 1). The SQL query does not include any ranking or ordering semantics.
- **q0:relation_semantics: S** — The query joins two tables on CDSCode, which is a key relationship, and filters based on Magnet = 1 and NumTstTakr > 500. This involves combining entities and grouping by the school entity (implied by the Magnet field).

## Question 6, candidate 4

Among the schools with the SAT test takers of over 500, please list the schools that are magnet schools or offer a magnet program.

Evidence: Magnet schools or offer a magnet program means that Magnet = 1

Benchmark execution label: False

```sql
SELECT s.CDSCode, s.Magnet FROM schools AS s INNER JOIN satscores AS ss ON s.CDSCode = ss.cds WHERE ss.NumTstTakr > 500 AND s.Magnet = 1
```

- **q0:output_semantics: S** — The SQL query selects the CDSCode and Magnet column from schools that have SAT test takers over 500 and are magnet schools. The question asks for schools with SAT test takers over 500 that are magnet schools or offer a magnet program (Magnet = 1). The query correctly identifies magnet schools (Magnet = 1) and includes the CDSCode, which is needed to list the schools. The projection and filtering align with the question's requirements.
- **q0:constraint_semantics: S** — The SQL query correctly applies the required filters: it ensures that schools have over 500 SAT test takers (NumTstTakr > 500) and are magnet schools (Magnet = 1). These conditions align with the question's requirements.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It simply asks to list schools with specific conditions (SAT test takers over 500 and Magnet = 1). The SQL query does not include any ranking or ordering semantics.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, which is a key relationship. It filters for schools with over 500 SAT test takers and checks if they are magnet schools (Magnet = 1). This satisfies the requirement to combine entities and group by the school entity (via the join and filter on Magnet).

## Question 7, candidate 0

What is the phone number of the school that has the highest number of test takers with an SAT score of over 1500?

Evidence: (none)

Benchmark execution label: True

```sql
SELECT s.Phone FROM schools AS s INNER JOIN satscores AS ss ON s.CDSCode = ss.cds WHERE ss.NumTstTakr > 0 ORDER BY ss.NumTstTakr DESC NULLS FIRST LIMIT 1
```

- **q0:output_semantics: U** — The SQL query selects the phone number of the school with the highest number of test takers (NumTstTakr) where NumTstTakr is greater than 0. The question asks for the phone number of the school with the highest number of test takers with an SAT score of over 1500. The query does not account for SAT scores, so it does not fully address the question.
- **q0:constraint_semantics: S** — The WHERE clause correctly filters for schools with test takers (NumTstTakr > 0), which is necessary to identify schools with test takers. The ordering by NumTstTakr DESC and limiting to 1 ensures the school with the highest number of test takers is selected. The Phone field is correctly projected to answer the question.
- **q0:ranking_semantics: S** — The question asks for the school with the highest number of test takers with an SAT score of over 1500, which requires finding the maximum value (extreme) and ordering by that value. The SQL uses ORDER BY ss.NumTstTakr DESC NULLS FIRST and LIMIT 1 to achieve this, satisfying the ranking semantics requirement.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, which is a key relationship. It then orders by NumTstTakr to find the school with the highest number of test takers, which requires grouping or sorting based on an entity (NumTstTakr) and understanding the relationship between the tables.

## Question 7, candidate 1

What is the phone number of the school that has the highest number of test takers with an SAT score of over 1500?

Evidence: (none)

Benchmark execution label: True

```sql
SELECT T2.Phone FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T1.NumGE1500 > 0 ORDER BY T1.NumGE1500 DESC NULLS FIRST LIMIT 1
```

- **q0:output_semantics: S** — The SQL query selects the phone number of the school with the highest number of test takers with an SAT score over 1500. The question asks for the phone number of the school with the highest number of such test takers, and the query correctly identifies this by ordering by NumGE1500 in descending order and limiting to 1. The projection (T2.Phone) matches the requested entity.
- **q0:constraint_semantics: S** — The WHERE clause correctly filters for schools with at least one test taker who scored over 1500 on the SAT (NumGE1500 > 0), which is required to identify the school with the highest number of such test takers. The ordering and limiting ensures the school with the highest count is selected.
- **q0:ranking_semantics: S** — The question asks for the school with the highest number of test takers with an SAT score over 1500, which requires finding the maximum value (extreme) and retrieving the corresponding phone number. The SQL uses ORDER BY with DESC and LIMIT 1 to achieve this, satisfying the ranking semantics requirement.
- **q0:relation_semantics: S** — The query joins the satscores and schools tables on CDSCode, which is a foreign key relationship. It then filters for schools with at least one test taker scoring over 1500 and orders by the count of such test takers to find the school with the highest count. This involves grouping by the school entity (via the join) and requires understanding the relationship between the tables.

## Question 7, candidate 2

What is the phone number of the school that has the highest number of test takers with an SAT score of over 1500?

Evidence: (none)

Benchmark execution label: True

```sql
SELECT T1.Phone FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.NumGE1500 = (SELECT MAX(NumGE1500) FROM satscores)
```

- **q0:output_semantics: S** — The SQL query selects the phone number of the school(s) with the highest number of test takers who scored over 1500 on the SAT. The question asks for the phone number of the school with the highest number of such test takers, and the query correctly identifies this by joining the schools and satscores tables and filtering for the maximum NumGE1500 value. The projection (Phone) matches the requested entity.
- **q0:constraint_semantics: S** — The WHERE clause correctly filters for the maximum NumGE1500 value from the satscores table, which is necessary to identify the school(s) with the highest number of test takers scoring over 1500. This satisfies the constraint.
- **q0:ranking_semantics: S** — The question asks for the school with the highest number of test takers with an SAT score of over 1500, which requires identifying the maximum value (extreme) and retrieving the corresponding school. The SQL uses a subquery with MAX(NumGE1500) to find the extreme value and joins with the schools table to get the phone number. This satisfies the ranking/extrema requirement.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, which is a key relationship. It then filters for the maximum NumGE1500 value, which requires understanding the relationship between the tables to correctly identify the school with the highest number of test takers with an SAT score over 1500.
- **q1:output_semantics: S** — The SQL query attempts to find the phone number of the school with the highest number of test takers with an SAT score of over 1500. The subquery correctly identifies the maximum value of NumGE1500 from the satscores table. However, the main query selects the Phone from schools, which is correct as it links the schools table with the satscores table via CDSCode. The aggregation and projection are appropriate for the task.
- **q1:constraint_semantics: S** — The subquery correctly computes the maximum value of NumGE1500 from the satscores table, which is necessary for identifying the school with the highest number of test takers with an SAT score of over 1500.
- **q1:ranking_semantics: S** — The question asks for the school with the highest number of test takers with an SAT score of over 1500, which requires identifying the maximum value (extreme) and retrieving the corresponding school's phone number. The SQL uses a subquery with MAX(NumGE1500) to find the extreme value, satisfying the ranking_semantics rubric.
- **q1:relation_semantics: S** — The target block retrieves the maximum value of NumGE1500 from the satscores table, which is necessary to identify the school(s) with the highest number of test takers scoring over 1500. This is part of the relationship requirement to find the correct school(s) and subsequently retrieve their phone numbers from the schools table.

## Question 7, candidate 3

What is the phone number of the school that has the highest number of test takers with an SAT score of over 1500?

Evidence: (none)

Benchmark execution label: True

```sql
SELECT Phone FROM schools WHERE CDSCode = (SELECT cds FROM satscores WHERE NumGE1500 = (SELECT MAX(NumGE1500) FROM satscores))
```

- **q0:output_semantics: S** — The SQL query correctly selects the phone number from the schools table where the CDSCode matches the school with the maximum number of test takers scoring over 1500 on the SAT. The logic aligns with the question's requirements.
- **q0:constraint_semantics: S** — The SQL query correctly filters the schools table to find the one with the highest number of test takers with an SAT score of over 1500 by using the subquery to get the maximum NumGE1500 value from the satscores table and matching it to the corresponding CDSCode. The constraint is satisfied as it correctly applies the required filtering.
- **q0:ranking_semantics: S** — The question asks for the school with the highest number of test takers with an SAT score of over 1500, which requires identifying the maximum value (extreme) and then retrieving the corresponding school's phone number. The SQL uses a subquery to find the maximum NumGE1500 and then selects the school with that value, satisfying the ranking/extrema requirement.
- **q0:relation_semantics: U** — The query requires joining the schools table with the satscores table to find the school with the highest number of test takers with an SAT score over 1500. The current query does not explicitly perform this join, but it uses a subquery to find the CDSCode with the maximum NumGE1500, which implies a relationship between the tables. However, the relationship is not explicitly defined through a JOIN clause, which is required under the rubric.
- **q1:output_semantics: S** — The SQL query correctly selects the phone number from the schools table where the CDSCode matches the school with the maximum number of test takers (NumGE1500) in the satscores table. The projection (Phone) and the logic to find the school with the highest NumGE1500 are aligned with the question's requirements.
- **q1:constraint_semantics: S** — The WHERE clause correctly filters for schools with the maximum number of test takers having an SAT score over 1500, as it uses the MAX(NumGE1500) from the satscores table.
- **q1:ranking_semantics: S** — The question asks for the school with the highest number of test takers with an SAT score of over 1500, which requires identifying the maximum value (extreme) and then retrieving the corresponding school. The SQL uses a subquery with MAX(NumGE1500) to find the maximum value and then selects the school with that value. This satisfies the ranking/extrema requirement.
- **q1:relation_semantics: S** — The query requires finding the school with the highest number of test takers with an SAT score over 1500, which involves grouping or relating entities. The subquery in the WHERE clause identifies the maximum NumGE1500 value, and the outer query retrieves the phone number of the corresponding school. This involves a relationship between the schools and satscores tables through the CDSCode.
- **q2:output_semantics: S** — The block correctly computes the maximum value of NumGE1500 from the satscores table, which is necessary to find the school with the highest number of test takers with an SAT score of over 1500. This aligns with the question's requirement.
- **q2:constraint_semantics: S** — The subquery correctly computes the maximum number of test takers with an SAT score of over 1500 using MAX(NumGE1500). This value is then used to filter the schools in the outer query, which is the required population for finding the school with the highest number of such test takers.
- **q2:ranking_semantics: S** — The question asks for the school with the highest number of test takers with an SAT score over 1500, which requires identifying the maximum value (extreme) and then finding the corresponding school. The SQL uses a subquery with MAX(NumGE1500) to find the extreme value, which satisfies the ranking_semantics rubric.
- **q2:relation_semantics: S** — The target block retrieves the maximum value of NumGE1500 from the satscores table, which is necessary to identify the school(s) with the highest number of test takers scoring over 1500. This is part of the relationship requirement to find the correct CDSCode in the schools table.

## Question 8, candidate 0

What is the number of SAT test takers of the schools with the highest FRPM count for K-12 students?

Evidence: (none)

Benchmark execution label: True

```sql
SELECT T2.NumTstTakr FROM frpm AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.CDSCode = (SELECT CDSCode FROM frpm ORDER BY "FRPM Count (K-12)" DESC LIMIT 1) LIMIT 1
```

- **q0:output_semantics: S** — The SQL query selects the number of SAT test takers (NumTstTakr) from the satscores table for the school with the highest FRPM Count (K-12). This directly answers the question by retrieving the required entity (number of SAT test takers) for the school with the highest FRPM count.
- **q0:constraint_semantics: S** — The SQL correctly filters for the school with the highest FRPM Count (K-12) by ordering the FRPM Count in descending order and limiting to 1, which aligns with the question's requirement to find the number of SAT test takers for the schools with the highest FRPM count.
- **q0:ranking_semantics: S** — The question asks for the number of SAT test takers from schools with the highest FRPM count, which requires identifying the top FRPM count and then retrieving the corresponding SAT test taker count. The SQL uses a subquery with ORDER BY and LIMIT 1 to find the highest FRPM count, which satisfies the ranking requirement.
- **q0:relation_semantics: S** — The query joins the frpm and satscores tables on CDSCode, which is a foreign key relationship. This satisfies the requirement of combining entities as the question asks for the number of SAT test takers for schools with the highest FRPM count.
- **q1:output_semantics: S** — The SQL query selects the CDSCode from the frpm table ordered by 'FRPM Count (K-12)' in descending order and limits to 1, which correctly identifies the school with the highest FRPM count. The main query then joins this result with the satscores table to retrieve the number of SAT test takers for that school. The projection is correct as it retrieves the required entity 'NumTstTakr'.
- **q1:constraint_semantics: S** — The subquery correctly orders the frpm table by 'FRPM Count (K-12)' in descending order and limits to the top result, which is the required population for schools with the highest FRPM count. This satisfies the constraint.
- **q1:ranking_semantics: S** — The question asks for the number of SAT test takers from schools with the highest FRPM count. The SQL block uses ORDER BY to find the highest FRPM count and LIMIT 1 to get the top result, which satisfies the ranking requirement.
- **q1:relation_semantics: S** — The block selects the CDSCode from the frpm table ordered by 'FRPM Count (K-12)' descending, which is necessary to identify the school(s) with the highest FRPM count. This involves a relationship between the frpm table and the satscores table through the CDSCode, which is used to retrieve the number of SAT test takers.

## Question 8, candidate 1

What is the number of SAT test takers of the schools with the highest FRPM count for K-12 students?

Evidence: (none)

Benchmark execution label: True

```sql
SELECT SUM(s.NumTstTakr) AS TotalSATTestTakers FROM frpm AS f INNER JOIN satscores AS s ON f.CDSCode = s.cds WHERE f."FRPM Count (K-12)" = (SELECT MAX("FRPM Count (K-12)") FROM frpm)
```

- **q0:output_semantics: C** — The SQL query attempts to sum the number of SAT test takers for schools with the highest FRPM count. However, the question asks for the number of SAT test takers of the schools with the highest FRPM count, not the total across all such schools. The SUM function is incorrect as it aggregates the values instead of selecting the individual values.
- **q0:constraint_semantics: S** — The SQL correctly filters for schools with the highest FRPM count (K-12) by using a subquery to select the maximum value and comparing it in the WHERE clause. This satisfies the constraint of identifying the required population based on the FRPM count.
- **q0:ranking_semantics: S** — The question asks for the number of SAT test takers of schools with the highest FRPM count. The SQL uses a subquery to find the maximum FRPM count and then sums the test takers for schools matching that value. This satisfies the ranking/extrema requirement as it identifies the top FRPM count and aggregates the relevant data.
- **q0:relation_semantics: S** — The query joins the frpm and satscores tables on CDSCode, which is a foreign key relationship. It then filters for the highest FRPM Count (K-12) and sums the SAT test takers, indicating a relationship requirement between entities.
- **q1:output_semantics: S** — The SQL query correctly identifies the maximum FRPM Count (K-12) from the frpm table and uses it to filter the schools with the highest FRPM count. The SUM(s.NumTstTakr) correctly calculates the total number of SAT test takers for those schools. The output semantics align with the question's request.
- **q1:constraint_semantics: S** — The subquery correctly identifies the maximum FRPM Count (K-12) value from the frpm table, which is necessary for filtering schools with the highest FRPM count. This aligns with the question's requirement to find the number of SAT test takers for those schools.
- **q1:ranking_semantics: S** — The question asks for the number of SAT test takers from schools with the highest FRPM count. The SQL uses a subquery to find the maximum FRPM count and filters schools with that value, which satisfies the ranking/extrema requirement. The MAX function is used correctly to identify the extreme value.
- **q1:relation_semantics: S** — The target block retrieves the maximum FRPM Count (K-12) from the frpm table, which is necessary to identify the schools with the highest FRPM count. This is a relational operation as it involves filtering based on a value derived from the frpm table, establishing a relationship between the frpm and satscores tables through the join condition.

## Question 8, candidate 2

What is the number of SAT test takers of the schools with the highest FRPM count for K-12 students?

Evidence: (none)

Benchmark execution label: True

```sql
SELECT T1.NumTstTakr FROM satscores AS T1 INNER JOIN frpm AS T2 ON T2.CDSCode = T1.cds ORDER BY T2."FRPM Count (K-12)" DESC LIMIT 1
```

- **q0:output_semantics: S** — The SQL query selects the number of SAT test takers (NumTstTakr) from the satscores table, joined with the frpm table on CDSCode. It orders the results by the FRPM Count (K-12) in descending order and limits the result to 1 row. This correctly retrieves the number of SAT test takers for the school with the highest FRPM count. The projection and target are aligned with the question.
- **q0:constraint_semantics: S** — The SQL query correctly filters and orders the data based on the 'FRPM Count (K-12)' column to find the school with the highest FRPM count and retrieves the number of SAT test takers for that school. The constraint on the FRPM count is properly applied in the ORDER BY clause.
- **q0:ranking_semantics: S** — The question asks for the number of SAT test takers of the schools with the highest FRPM count. The SQL query orders by FRPM Count (K-12) in descending order and limits to 1, which satisfies the requirement for the top school with the highest FRPM count. The result would be the number of SAT test takers for that school.
- **q0:relation_semantics: S** — The query joins the satscores and frpm tables on CDSCode, which is a foreign key relationship. It then orders by the FRPM Count (K-12) in descending order and limits to the top result. This satisfies the requirement of combining entities and grouping by an entity (implicitly through ordering and limiting).

## Question 8, candidate 4

What is the number of SAT test takers of the schools with the highest FRPM count for K-12 students?

Evidence: (none)

Benchmark execution label: True

```sql
SELECT T3.NumTstTakr FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode INNER JOIN satscores AS T3 ON T1.CDSCode = T3.cds WHERE T2."FRPM Count (K-12)" = (SELECT MAX("FRPM Count (K-12)") FROM frpm)
```

- **q0:output_semantics: S** — The SQL query selects the number of SAT test takers (NumTstTakr) from schools that have the highest FRPM Count (K-12). The question asks for the number of SAT test takers of the schools with the highest FRPM count, which matches the projection and filtering in the query. The output semantics are correct.
- **q0:constraint_semantics: S** — The SQL correctly filters for schools with the highest FRPM count (K-12) by using the MAX function on the FRPM Count (K-12) column. This satisfies the constraint of identifying the required population based on the highest FRPM count.
- **q0:ranking_semantics: S** — The question asks for the number of SAT test takers of schools with the highest FRPM count. The SQL uses a subquery to find the maximum FRPM count and selects schools with that value, which satisfies the ranking/extrema requirement. There is no ordering or limit specified in the main query, but the subquery correctly identifies the extremum.
- **q0:relation_semantics: S** — The query joins schools, frpm, and satscores tables to find the number of SAT test takers for schools with the highest FRPM count. This involves combining entities and grouping by the school entity to retrieve the required data.
- **q1:output_semantics: S** — The SQL computes the maximum FRPM Count (K-12) from the frpm table, which is used as a filter to select schools with the highest FRPM count. This aligns with the question's requirement to find the number of SAT test takers for those schools. The aggregation and projection are correct for the task.
- **q1:constraint_semantics: S** — The subquery correctly computes the maximum FRPM Count (K-12) from the frpm table, which is necessary to identify schools with the highest FRPM count. This satisfies the constraint as it properly filters the required population.
- **q1:ranking_semantics: S** — The question asks for the number of SAT test takers of schools with the highest FRPM count. The SQL uses a subquery with MAX() to find the highest FRPM count, which is an extremum. This satisfies the ranking_semantics rubric as it identifies the maximum value, which is an extreme value.
- **q1:relation_semantics: S** — The target block selects the maximum FRPM Count (K-12) from the frpm table, which is necessary to identify the schools with the highest FRPM count. This is part of the relationship requirement to filter schools based on their FRPM count before retrieving the number of SAT test takers.

## Question 9, candidate 0

Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?

Evidence: (none)

Benchmark execution label: False

```sql
SELECT COUNT(s.CDSCode) AS Total_Schools FROM schools AS s INNER JOIN satscores AS ss ON s.CDSCode = ss.cds WHERE ss.AvgScrMath > 560 AND s.FundingType = 'Direct Charter Funding'
```

- **q0:output_semantics: S** — The SQL query counts schools with an average Math score over 560 and a FundingType of 'Direct Charter Funding', which directly answers the question. The aggregate function COUNT is appropriate for counting the number of schools.
- **q0:constraint_semantics: S** — The WHERE clause correctly filters schools with an average Math score over 560 and checks for 'Direct Charter Funding' in the FundingType column, which aligns with the question's requirement.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It asks for a count of schools meeting specific criteria, which does not involve ranking semantics.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, groups by nothing, and filters based on average math score and funding type. This satisfies the requirement to combine entities (schools and satscores) and check for charter-funded schools with average math scores over 560.

## Question 9, candidate 1

Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?

Evidence: (none)

Benchmark execution label: False

```sql
SELECT COUNT(s.CDSCode) FROM schools AS s INNER JOIN satscores AS st ON s.CDSCode = st.cds WHERE st.AvgScrMath > 560 AND s.FundingType = 'Directly Charter-Funded'
```

- **q0:output_semantics: S** — The SQL query counts the number of schools with an average Math score over 560 and a FundingType of 'Directly Charter-Funded'. This matches the question's request for the count of schools directly charter-funded with an average Math score over 560.
- **q0:constraint_semantics: S** — The SQL query correctly applies the required filters: it checks for schools with an average Math score over 560 (st.AvgScrMath > 560) and filters for schools that are directly charter-funded (s.FundingType = 'Directly Charter-Funded'). These conditions align with the question's requirements.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It asks for a count of schools meeting specific criteria, which does not involve ranking semantics.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, filters by average math score and funding type, and counts the number of schools. This involves grouping by an entity (schools) and requires understanding the relationship between the tables.

## Question 9, candidate 2

Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?

Evidence: (none)

Benchmark execution label: False

```sql
SELECT COUNT(T1.cds) FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrMath > 560 AND T2.FundingType = 'Charter'
```

- **q0:output_semantics: U** — The SQL query counts the number of schools with an average Math score over 560 and a FundingType of 'Charter'. The question asks for the number of schools directly charter-funded, which aligns with the query's logic. However, the query does not explicitly check if the school is directly charter-funded, as the 'FundingType' field may not directly indicate this. The rubric requires checking the requested entities/values, and the query's use of 'FundingType = 'Charter'' may not fully capture the intended meaning of 'directly charter-funded'.
- **q0:constraint_semantics: S** — The SQL query correctly filters schools with an average Math score over 560 and checks if they are directly charter-funded by using the FundingType column. The WHERE clause applies the required constraints as per the question.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It asks for a count of schools meeting specific criteria, which does not involve ranking semantics.
- **q0:relation_semantics: S** — The query joins the satscores and schools tables on cds/CDSCode, which establishes a relationship between the entities. It filters for schools with an average math score over 560 and Charter FundingType, then counts the number of such schools. This satisfies the requirement of combining entities and grouping by an entity (FundingType).

## Question 9, candidate 3

Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?

Evidence: (none)

Benchmark execution label: False

```sql
SELECT COUNT(T1.CDSCode) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode INNER JOIN satscores AS T3 ON T1.CDSCode = T3.cds WHERE T3.AvgScrMath > 560 AND T2.FundingType = 'Charter'
```

- **q0:output_semantics: S** — The SQL query counts the number of schools with an average Math score over 560 and a FundingType of 'Charter'. The question asks for the number of schools directly charter-funded, which aligns with the FundingType filter. The COUNT function is appropriate for counting the number of schools.
- **q0:constraint_semantics: S** — The SQL query correctly filters schools with an average Math score over 560 and checks if they are directly charter-funded by using the FundingType column from the schools table. The constraint is satisfied as it aligns with the question's requirements.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It asks for a count of schools meeting specific criteria, which does not involve ranking semantics.
- **q0:relation_semantics: S** — The query joins three tables (frpm, schools, satscores) and filters based on SAT math scores and funding type. It requires grouping or relationships between entities to answer the question about schools directly charter-funded with an average math score over 560.

## Question 9, candidate 4

Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?

Evidence: (none)

Benchmark execution label: False

```sql
SELECT COUNT(DISTINCT T1.CDSCode) FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.AvgScrMath > 560 AND T1.FundingType = 'Direct Charter'
```

- **q0:output_semantics: S** — The SQL query counts the number of distinct schools with an average Math score over 560 and a FundingType of 'Direct Charter', which directly answers the question. The aggregate function COUNT(DISTINCT) is appropriate for counting unique schools, and the conditions match the question's requirements.
- **q0:constraint_semantics: S** — The SQL query correctly filters schools with an average Math score over 560 and checks if they are directly charter-funded by using the FundingType column. The WHERE clause applies the required constraints on both the SAT score and the funding type.
- **q0:ranking_semantics: NA** — The question does not request an extreme, top/bottom k, ordering, or rank. It asks for a count of schools meeting specific criteria, which does not involve ranking semantics.
- **q0:relation_semantics: S** — The query joins the schools and satscores tables on CDSCode, which is a key relationship. It filters based on average math scores and funding type, which requires understanding the relationship between entities to correctly count schools that meet the criteria.

