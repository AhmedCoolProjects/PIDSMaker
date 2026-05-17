## 1. PIDSMaker Postgress Databases

| File                 | Size      |
| -------------------- | --------- |
| `clearscope_e3.dump` | 588 MB    |
| `theia_e3.dump`      | 1.1 GB    |
| `cadets_e3.dump`     | 1.4 GB    |
| `optc_501.dump`      | 1.5 GB    |
| `optc_051.dump`      | 1.7 GB    |
| `optc_201.dump`      | 2.0 GB    |
| `theia_e5.dump`      | 5.8 GB    |
| `clearscope_e5.dump` | 6.2 GB    |
| `cadets_e5.dump`     | **36 GB** |

> for first experiments use these datasets:
> - `clearscope_e3.dump`
> - `theia_e3.dump`
> - `cadets_e3.dump`
> - `optc_501.dump`

## 2. Tables in all databases

```sql
// postgres=# 
\c clearscope_e3;
```

```sql
// clearscope_e3=# 
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
```

> (4 rows)

 | table_name         |
 | ------------------ |
 | event_table        |
 | file_node_table    |
 | netflow_node_table |
 | subject_node_table |


## 3. Columns use in all tables

clearscope_e3=# 
```sql
SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;
```

> (24 rows)

| table_name         | column_name   | data_type         |
| ------------------ | ------------- | ----------------- |
| event_table        | src_node      | character varying |
| event_table        | src_index_id  | character varying |
| event_table        | operation     | character varying |
| event_table        | dst_node      | character varying |
| event_table        | dst_index_id  | character varying |
| event_table        | event_uuid    | character varying |
| event_table        | timestamp_rec | bigint            |
| event_table        | _id           | bigint            |
| file_node_table    | node_uuid     | character varying |
| file_node_table    | hash_id       | character varying |
| file_node_table    | path          | character varying |
| file_node_table    | index_id      | bigint            |
| netflow_node_table | node_uuid     | character varying |
| netflow_node_table | hash_id       | character varying |
| netflow_node_table | src_addr      | character varying |
| netflow_node_table | src_port      | character varying |
| netflow_node_table | dst_addr      | character varying |
| netflow_node_table | dst_port      | character varying |
| netflow_node_table | index_id      | bigint            |
| subject_node_table | node_uuid     | character varying |
| subject_node_table | hash_id       | character varying |
| subject_node_table | path          | character varying |
| subject_node_table | cmd           | character varying |
| subject_node_table | index_id      | bigint            |

```bash
# prompt
I have my postgress instance (I already did run it in my db folder like this bash pg-start2.sh) that we can access with this command (apptainer exec instance://postgres_instance2 psql -h localhost -p 5433 -U postgres). There you have access to all databases (clearscope_e3, theia_e3, cadets_e3, optc_501, optc_051, optc_201, theia_e5, clearscope_e5, cadets_e5) and in each database you have 4 tables (event_table, file_node_table, netflow_node_table, subject_node_table) and each table has some columns that we can use to do some statistics and understand better the data. So let's start by doing some statistics on the data to understand it better. We can start by doing some statistics on the 'EVENTS' table in all databases and then we can do some statistics on the 'File Objects', 'Netflow Objects' and 'Subject Objects' tables in all databases. This will help us to understand better the data and also to get some insights that can help us. List what statistics you think we can do and then let's see the plan before implementing it.

write the results of the statistics in a .md file called pidsmaker-darpa-statistics.md and also write the SQL queries that we used to do the statistics in a .sql file called pidsmaker-darpa-statistics.sql. I want to have sections like this (Statistics of 'EVENTS' in all databases, Statistics of 'File Objects' in all databases, Statistics of 'Netflow Objects' in all databases, Statistics of 'Subject Objects' in all databases) and in each section I want to have the results of the statistics and also the SQL queries that we used to do the statistics. I want to have a clear and organized structure for the .md file and also for the .sql file.
```

# DARPA CADETS/OPTC/Theia/ClearScope — Database Statistics

> Generated from PostgreSQL instance 2 (`postgres_instance2`, port 5433).
> 9 databases, 4 tables each (event_table, file_node_table, netflow_node_table, subject_node_table).

---

## 1. Statistics of 'Events' (`event_table`) in all databases

### 1.1 Total Event Count per Database

| Database      |  Total Events |
| ------------- | ------------: |
| cadets_e3     |    36,484,667 |
| cadets_e5     | 1,075,370,668 |
| clearscope_e3 |    18,316,129 |
| clearscope_e5 |   198,794,211 |
| theia_e3      |    44,366,117 |
| theia_e5      |   140,994,662 |
| optc_051      |    31,611,029 |
| optc_201      |    38,151,059 |
| optc_501      |    28,049,214 |

```sql
SELECT COUNT(*) FROM event_table;
```

### 1.2 Time Range per Database

| Database      | Min Timestamp (approx) | Max Timestamp (approx) | Span     |
| ------------- | ---------------------- | ---------------------- | -------- |
| cadets_e3     | 2018-04-02             | 2018-04-13             | ~12 days |
| cadets_e5     | 2019-05-07             | 2019-05-17             | ~10 days |
| clearscope_e3 | 2018-04-02             | 2018-04-13             | ~12 days |
| clearscope_e5 | 2019-05-07             | 2019-05-20             | ~14 days |
| theia_e3      | 2018-04-03             | 2018-04-13             | ~10 days |
| theia_e5      | 2019-05-07             | 2019-05-17             | ~10 days |
| optc_051      | 2019-09-17             | 2019-09-25             | ~9 days  |
| optc_201      | 2019-09-17             | 2019-09-25             | ~9 days  |
| optc_501      | 2019-09-19             | 2019-09-25             | ~7 days  |

```sql
SELECT MIN(timestamp_rec), MAX(timestamp_rec) FROM event_table;
```

### 1.3 Operation Distribution per Database

#### cadets_e3 (25 operation types)
| Operation              |      Count |
| ---------------------- | ---------: |
| EVENT_READ             | 12,527,239 |
| EVENT_CLOSE            |  6,124,156 |
| EVENT_MMAP             |  4,648,096 |
| EVENT_LSEEK            |  4,016,402 |
| EVENT_OPEN             |  3,843,324 |
| EVENT_WRITE            |  1,602,681 |
| EVENT_MODIFY_PROCESS   |  1,099,007 |
| EVENT_CREATE_OBJECT    |    940,967 |
| EVENT_CHANGE_PRINCIPAL |    304,919 |
| EVENT_FORK             |    223,311 |

#### cadets_e5 (25 operation types)
| Operation            |       Count |
| -------------------- | ----------: |
| EVENT_READ           | 304,289,795 |
| EVENT_WRITE          | 299,784,434 |
| EVENT_CLOSE          | 136,165,614 |
| EVENT_OPEN           | 133,125,446 |
| EVENT_MMAP           |  82,137,773 |
| EVENT_MODIFY_PROCESS |  60,558,036 |
| EVENT_LSEEK          |  23,118,769 |
| EVENT_UNLINK         |   6,313,086 |
| EVENT_FORK           |   5,805,366 |
| EVENT_EXIT           |   5,771,790 |

#### clearscope_e3 (8 operation types)
| Operation      |     Count |
| -------------- | --------: |
| EVENT_READ     | 9,649,225 |
| EVENT_WRITE    | 6,056,247 |
| EVENT_OPEN     |   794,935 |
| EVENT_RECVFROM |   463,906 |
| EVENT_SENDTO   |   455,852 |
| EVENT_SENDMSG  |   429,403 |
| EVENT_RECVMSG  |   379,417 |
| EVENT_CONNECT  |    87,144 |

#### clearscope_e5 (10 operation types)
| Operation      |       Count |
| -------------- | ----------: |
| EVENT_READ     | 120,957,956 |
| EVENT_OPEN     |  38,836,671 |
| EVENT_WRITE    |  38,700,519 |
| EVENT_RECVFROM |      92,692 |
| EVENT_SENDMSG  |      64,053 |
| EVENT_RECVMSG  |      62,842 |
| EVENT_CONNECT  |      44,071 |
| EVENT_SENDTO   |      34,735 |
| EVENT_CLONE    |         644 |
| EVENT_EXECUTE  |          28 |

#### optc_051 (10 operation types)
| Operation |      Count |
| --------- | ---------: |
| START     | 17,406,744 |
| MESSAGE   |  7,092,481 |
| OPEN      |  2,665,055 |
| MODIFY    |  1,622,741 |
| READ      |  1,133,543 |
| WRITE     |  1,002,176 |
| CREATE    |    406,612 |
| RENAME    |    150,184 |
| DELETE    |     86,631 |
| TERMINATE |     44,862 |

#### optc_201 (10 operation types)
| Operation |      Count |
| --------- | ---------: |
| START     | 18,879,853 |
| MESSAGE   |  7,820,965 |
| OPEN      |  6,168,599 |
| MODIFY    |  1,714,316 |
| READ      |  1,581,012 |
| WRITE     |  1,125,713 |
| CREATE    |    497,026 |
| RENAME    |    167,090 |
| DELETE    |    113,364 |
| TERMINATE |     83,121 |

#### optc_501 (10 operation types)
| Operation |      Count |
| --------- | ---------: |
| START     | 14,887,961 |
| MESSAGE   |  6,119,671 |
| OPEN      |  2,980,542 |
| MODIFY    |  1,394,014 |
| READ      |  1,118,299 |
| WRITE     |    891,207 |
| CREATE    |    383,825 |
| RENAME    |    132,362 |
| DELETE    |     81,324 |
| TERMINATE |     60,009 |

#### theia_e3 (15 operation types)
| Operation                |      Count |
| ------------------------ | ---------: |
| EVENT_RECVFROM           | 14,216,098 |
| EVENT_READ               |  8,363,322 |
| EVENT_OPEN               |  6,606,629 |
| EVENT_READ_SOCKET_PARAMS |  3,894,694 |
| EVENT_WRITE              |  3,678,040 |
| EVENT_RECVMSG            |  3,030,523 |
| EVENT_SENDMSG            |  2,407,595 |
| EVENT_SENDTO             |    635,486 |
| EVENT_CONNECT            |    591,943 |
| EVENT_UNLINK             |    410,061 |

#### theia_e5 (16 operation types)
| Operation      |      Count |
| -------------- | ---------: |
| EVENT_WRITE    | 66,155,235 |
| EVENT_OPEN     | 38,130,199 |
| EVENT_READ     | 29,644,315 |
| EVENT_RECVFROM |  1,775,374 |
| EVENT_RECVMSG  |  1,531,703 |
| EVENT_EXIT     |  1,235,348 |
| EVENT_CLONE    |  1,205,496 |
| EVENT_EXECUTE  |    996,631 |
| EVENT_SENDTO   |    165,338 |
| EVENT_CONNECT  |    104,409 |

```sql
SELECT operation, COUNT(*) as cnt FROM event_table GROUP BY operation ORDER BY cnt DESC;
```

### 1.4 Operation Category Summary

| Database      |   Read/Open |  Open/Close | Write/Modify | Network/FS Mgmt | Process/Lifecycle |     Other |
| ------------- | ----------: | ----------: | -----------: | --------------: | ----------------: | --------: |
| cadets_e3     |  12,721,101 |   9,967,480 |    8,534,612 |       4,303,975 |           957,499 |         — |
| cadets_e5     | 308,336,844 | 269,291,060 |  446,931,739 |      31,531,827 |        19,279,198 |         — |
| clearscope_e3 |  10,492,548 |     794,935 |    6,941,502 |          87,144 |                 — |         — |
| clearscope_e5 | 121,113,490 |  38,836,671 |   38,799,307 |          44,071 |                28 |       644 |
| optc_051      |   3,798,598 |           — |    3,031,529 |         236,815 |        17,451,606 | 7,092,481 |
| optc_201      |   7,749,611 |           — |    3,337,055 |         280,454 |        18,962,974 | 7,820,965 |
| optc_501      |   4,098,841 |           — |    2,669,046 |         213,686 |        14,947,970 | 6,119,671 |
| theia_e3      |  25,609,943 |   6,606,629 |    6,721,279 |       1,002,004 |           116,621 | 4,309,641 |
| theia_e5      |  32,951,392 |  38,130,199 |   66,350,580 |         110,420 |         2,246,568 | 1,205,503 |

```sql
SELECT CASE 
  WHEN operation IN ('EVENT_READ','EVENT_RECVFROM','EVENT_RECVMSG','READ','OPEN') THEN 'Read/Open'
  WHEN operation IN ('EVENT_WRITE','EVENT_SENDMSG','EVENT_SENDTO','EVENT_CREATE_OBJECT',
    'EVENT_MODIFY_FILE_ATTRIBUTES','EVENT_MODIFY_PROCESS','EVENT_MMAP','WRITE','MODIFY','CREATE') THEN 'Write/Modify'
  WHEN operation IN ('EVENT_CONNECT','EVENT_ACCEPT','EVENT_BIND','EVENT_RENAME','EVENT_LINK',
    'EVENT_UNLINK','EVENT_TRUNCATE','EVENT_LSEEK','RENAME','DELETE') THEN 'Network/Filesystem Mgmt'
  WHEN operation IN ('EVENT_EXECUTE','EVENT_EXIT','EVENT_FORK','EVENT_SIGNAL',
    'EVENT_CHANGE_PRINCIPAL','START','TERMINATE') THEN 'Process/Lifecycle'
  WHEN operation IN ('EVENT_CLOSE','EVENT_OPEN','CLOSE') THEN 'Open/Close'
  ELSE 'Other'
END as category, COUNT(*) as cnt
FROM event_table GROUP BY category ORDER BY cnt DESC;
```

### 1.5 Top src_nodes per Database (Top 5)

| Database      | src_node (hash/uuid) | Event Count |
| ------------- | -------------------- | ----------: |
| cadets_e3     | 397b7aa...c1bd       |   7,353,245 |
|               | 40e7a16...6f33       |   1,646,395 |
|               | d482c2c...90cd       |   1,469,438 |
|               | 1af0ed1...14e3       |   1,306,970 |
|               | b51a562...86dd       |   1,033,349 |
| cadets_e5     | 34aabb7...573b       |  53,187,780 |
|               | 0a38e5a...6e6f5      |  38,364,147 |
|               | 36be71f...8287c      |  26,709,423 |
|               | eba9368...2ce8       |  17,264,814 |
|               | 0dd8786...7e562      |  17,239,072 |
| clearscope_e3 | 90d5b0a...cc0a       |   1,861,388 |
|               | 2a6a0c2...96aa       |   1,738,499 |
|               | 9ebf362...5ef3       |   1,604,790 |
|               | e57a615...ee2a       |   1,063,717 |
|               | b0c2f39...89e        |     898,180 |
| clearscope_e5 | 7248840...7437       |  20,545,187 |
|               | 568d8f0...9029       |  18,705,563 |
|               | e091df9...9252       |  17,126,978 |
|               | 71cea69...c8a8       |   9,257,768 |
|               | 771fe06...7f2        |   8,100,613 |
| optc_051      | 408ec09...2ecb       |     819,443 |
|               | e6ce56f...851f       |     797,007 |
|               | eb0c9a8...6fcb       |     539,962 |
|               | cd90e67...9136       |     243,941 |
|               | 2f046d8...79ae       |     242,746 |
| optc_201      | 70b6a61...0d6fe      |     862,715 |
|               | fad56a0...3c95       |     806,095 |
|               | 104cecd...a208       |     540,572 |
|               | 89f8312...4acd       |     465,729 |
|               | 6f38b86...6656       |     457,264 |
| optc_501      | 33cadfb...34a8       |     799,047 |
|               | ab16baa...0aa2       |     690,262 |
|               | fab2e62...91d1       |     538,475 |
|               | 9d0acaf...9fef       |     408,885 |
|               | 169f9f7...c66a       |     331,614 |
| theia_e3      | 3f9fabe...54f9b      |  12,921,969 |
|               | e3fed40...daf4a      |   3,614,099 |
|               | 051547d...1623       |   2,438,830 |
|               | 7027dec...adf8e      |     521,819 |
|               | 03fc119...61090      |     521,230 |
| theia_e5      | d4b41ef...0ab1       |  47,109,132 |
|               | c6a8632...02f7       |   3,977,094 |
|               | f735e36...3ed6       |   3,848,848 |
|               | f8f0d07...cd9a       |   2,188,111 |
|               | 0556d43...0928e      |   2,041,537 |

```sql
SELECT src_node, COUNT(*) as cnt FROM event_table 
GROUP BY src_node ORDER BY cnt DESC LIMIT 10;
```

### 1.6 Top dst_nodes per Database (Top 5)

| Database      | dst_node (hash/uuid) | Event Count |
| ------------- | -------------------- | ----------: |
| cadets_e3     | 8efa716...1c5b5      |     210,941 |
|               | d482c2c...90cd       |     210,918 |
|               | a175028...8460       |     210,905 |
|               | 37eb8a2...182d       |     201,719 |
|               | b45a110...5c164      |     172,007 |
| clearscope_e3 | 2a6a0c2...96aa       |   1,732,606 |
|               | 90d5b0a...cc0a       |   1,635,949 |
|               | 9ebf362...5ef3       |   1,490,547 |
|               | 7f6a213...d495       |     858,360 |
|               | 3857ad2...63f2       |     856,526 |
| clearscope_e5 | e091df9...9252       |  29,668,200 |
|               | 71cea69...c8a8       |  16,284,296 |
|               | 7248840...7437       |  13,703,187 |
|               | 568d8f0...9029       |  12,498,295 |
|               | d749bb4...229d       |  10,329,138 |
| optc_051      | feb7701...475d       |   8,605,225 |
|               | 94652ae...6dde       |   2,511,875 |
|               | 408ec09...2ecb       |   2,323,895 |
|               | dfcbb95...01c        |   2,138,778 |
|               | 90747cd...e015       |   1,993,856 |
| optc_201      | 359a1a8...dbfc       |   8,504,202 |
|               | a178a61...6f0e0      |   3,985,538 |
|               | fad56a0...3c95       |   2,325,491 |
|               | c80c6e1...2830       |   2,150,110 |
|               | 02e5f30...6af0       |   2,013,651 |
| optc_501      | 6d21f14...dfc6       |   8,593,470 |
|               | 33cadfb...34a8       |   2,341,621 |
|               | 2b5dd07...83f        |   2,124,952 |
|               | 387c24a...1bc8       |   1,985,677 |
|               | bde3d89...1970       |   1,915,128 |
| theia_e3      | 99e9147...caf5       |   3,555,613 |
|               | e2e80d5...4ab5e      |   3,535,484 |
|               | 2445264...67ff       |   2,446,779 |
|               | a750750...cdb0       |   2,140,591 |
|               | 3f9fabe...54f9b      |   2,002,423 |
| theia_e5      | 037e1df...dad7       |  47,109,333 |
|               | 408e355...be0d       |   4,309,600 |
|               | c6a8632...02f7       |   4,066,734 |
|               | 50ad41b...44eaf      |   2,268,269 |
|               | ecf5194...91b5       |   2,197,957 |

```sql
SELECT dst_node, COUNT(*) as cnt FROM event_table 
GROUP BY dst_node ORDER BY cnt DESC LIMIT 10;
```

### 1.7 Event Count by Day

#### cadets_e3
| Day        |    Events |
| ---------- | --------: |
| 2018-04-02 |    76,227 |
| 2018-04-03 | 2,770,927 |
| 2018-04-04 | 2,885,604 |
| 2018-04-05 | 3,132,540 |
| 2018-04-06 | 3,479,429 |
| 2018-04-07 | 4,168,064 |
| 2018-04-08 | 3,828,290 |
| 2018-04-09 | 3,284,696 |
| 2018-04-10 | 3,037,406 |
| 2018-04-11 | 3,310,145 |
| 2018-04-12 | 3,262,035 |
| 2018-04-13 | 3,249,304 |

#### cadets_e5
| Day        |      Events |
| ---------- | ----------: |
| 2019-05-07 |  37,011,175 |
| 2019-05-08 |  82,673,878 |
| 2019-05-09 |  78,254,569 |
| 2019-05-10 | 147,063,842 |
| 2019-05-11 |  87,593,008 |
| 2019-05-12 |  97,995,970 |
| 2019-05-13 |  76,448,735 |
| 2019-05-14 |  98,606,387 |
| 2019-05-15 | 120,278,824 |
| 2019-05-16 | 125,885,410 |
| 2019-05-17 | 123,558,870 |

#### clearscope_e3
| Day        |    Events |
| ---------- | --------: |
| 2018-04-02 |    40,258 |
| 2018-04-03 | 1,191,732 |
| 2018-04-04 | 1,462,046 |
| 2018-04-05 |   935,582 |
| 2018-04-06 | 1,083,499 |
| 2018-04-07 | 1,946,919 |
| 2018-04-08 | 1,111,903 |
| 2018-04-09 | 1,826,845 |
| 2018-04-10 | 3,054,453 |
| 2018-04-11 | 1,937,664 |
| 2018-04-12 | 2,102,869 |
| 2018-04-13 | 1,503,412 |

#### clearscope_e5
| Day        |     Events |
| ---------- | ---------: |
| 2019-05-07 | 24,363,346 |
| 2019-05-08 | 35,943,412 |
| 2019-05-09 | 34,119,633 |
| 2019-05-10 |  7,069,468 |
| 2019-05-11 |    518,818 |
| 2019-05-12 |    502,334 |
| 2019-05-13 |  6,433,822 |
| 2019-05-14 | 15,032,966 |
| 2019-05-15 | 13,290,274 |
| 2019-05-16 | 27,413,930 |
| 2019-05-17 | 24,134,454 |
| 2019-05-18 |  4,104,804 |
| 2019-05-19 |  3,881,011 |
| 2019-05-20 |  1,985,939 |

#### optc_051
| Day        |    Events |
| ---------- | --------: |
| 2019-09-17 | 2,000,625 |
| 2019-09-18 | 2,495,880 |
| 2019-09-19 | 4,768,950 |
| 2019-09-20 | 3,208,665 |
| 2019-09-21 | 4,139,038 |
| 2019-09-22 | 4,107,024 |
| 2019-09-23 | 4,067,211 |
| 2019-09-24 | 3,755,753 |
| 2019-09-25 | 3,067,883 |

#### optc_201
| Day        |    Events |
| ---------- | --------: |
| 2019-09-17 | 6,577,149 |
| 2019-09-18 | 4,134,791 |
| 2019-09-19 | 5,055,013 |
| 2019-09-20 | 3,174,727 |
| 2019-09-21 | 4,160,882 |
| 2019-09-22 | 4,261,741 |
| 2019-09-23 | 4,109,999 |
| 2019-09-24 | 3,735,341 |
| 2019-09-25 | 2,941,416 |

#### optc_501
| Day        |    Events |
| ---------- | --------: |
| 2019-09-19 | 5,034,335 |
| 2019-09-20 | 3,106,791 |
| 2019-09-21 | 4,340,487 |
| 2019-09-22 | 4,248,268 |
| 2019-09-23 | 4,213,681 |
| 2019-09-24 | 3,944,754 |
| 2019-09-25 | 3,160,898 |

#### theia_e3
| Day        |    Events |
| ---------- | --------: |
| 2018-04-03 | 7,338,104 |
| 2018-04-04 | 8,755,963 |
| 2018-04-05 | 2,508,708 |
| 2018-04-09 |   712,059 |
| 2018-04-10 | 5,625,415 |
| 2018-04-11 | 7,251,248 |
| 2018-04-12 | 6,900,759 |
| 2018-04-13 | 5,273,861 |

#### theia_e5
| Day        |     Events |
| ---------- | ---------: |
| 2019-05-07 | 55,440,420 |
| 2019-05-08 | 12,196,360 |
| 2019-05-09 |  5,553,124 |
| 2019-05-10 |  7,441,616 |
| 2019-05-11 |  7,008,828 |
| 2019-05-12 |  3,378,054 |
| 2019-05-13 |  3,167,731 |
| 2019-05-14 | 13,478,907 |
| 2019-05-15 | 12,612,026 |
| 2019-05-16 | 12,465,341 |
| 2019-05-17 |  8,252,255 |

```sql
SELECT DATE(TO_TIMESTAMP(timestamp_rec / 1000000000)) as day, COUNT(*) as cnt
FROM event_table WHERE timestamp_rec > 1000000000000000000
GROUP BY day ORDER BY day;
```

---

## 2. Statistics of 'File Objects' (`file_node_table`) in all databases

### 2.1 Total File Node Count & Unique File (hash_id) Count

| Database      | File Nodes | Unique hash_id | Null Path |
| ------------- | ---------: | -------------: | --------: |
| cadets_e3     |  2,303,164 |      2,303,164 | 2,260,767 |
| cadets_e5     |  8,434,050 |      8,434,050 | 4,909,211 |
| clearscope_e3 |     76,938 |         76,938 |         0 |
| clearscope_e5 |    324,210 |        324,210 |         0 |
| optc_051      |    380,963 |         56,328 |         1 |
| optc_201      |    472,501 |         69,139 |         2 |
| optc_501      |    345,365 |         56,899 |         1 |
| theia_e3      |    793,899 |        793,899 |         0 |
| theia_e5      |    984,937 |        984,937 |         0 |

Observations:
- **cadets** and **theia** datasets: every file node has a unique hash_id (one-to-one mapping, no content dedup).
- **optc** datasets: multiple file nodes share the same hash_id (≈6.8:1 ratio), implying repeated file accesses.
- **cadets** has many null/empty paths (~98% in cadets_e3, ~58% in cadets_e5).

```sql
SELECT COUNT(*) as total, COUNT(DISTINCT hash_id) as unique_hashes,
       COUNT(*) FILTER (WHERE path IS NULL OR path = '') as null_path
FROM file_node_table;
```

### 2.2 Top Paths per Database (Top 5)

#### cadets_e3
| Path                  | Count |
| --------------------- | ----: |
| /var/mail/bob.lock    | 4,615 |
| /var/mail/george.lock | 2,194 |
| /var/mail/frank.lock  | 2,057 |
| /tmp/.31.29ada        | 1,727 |
| /home/bob/Sent.lock   | 1,552 |

#### cadets_e5
| Path                           |  Count |
| ------------------------------ | -----: |
| /home/user/mail/sent-mail.lock | 13,890 |
| /usr/home/user/eraseme         |  5,075 |
| /tmp/.3c.3c4a                  |  2,670 |
| /usr/home/user/test/yf.c       |  2,572 |
| /usr/home/user/test/xw.c       |  2,565 |

#### optc_051
| Path                                                                  | Count |
| --------------------------------------------------------------------- | ----: |
| /Device/HarddiskVolume1/Windows/SysWOW64/ncr.key                      |   201 |
| /Device/HarddiskVolume1/Windows/System32/Tasks/...                    |   201 |
| /Device/HarddiskVolume1/ncr/kickoff/input/usernames.csv               |   201 |
| /Device/HarddiskVolume1/Python27/lib/site-packages/pyaes/__init__.pyc |   193 |
| /Device/HarddiskVolume1/Python27/lib/site-packages/pyaes/aes.pyc      |   193 |

#### theia_e3
| Path                                                        |  Count |
| ----------------------------------------------------------- | -----: |
| /var/lib/postgresql/9.1/main/pg_stat_tmp/pgstat.tmp         | 20,412 |
| /home/admin/.mozilla/firefox/...sessionCheckpoints.json.tmp |  7,820 |
| /home/admin/.mozilla/firefox/...recovery.js.tmp             |  7,493 |
| /run/ConsoleKit/database~                                   |  3,779 |
| /home/admin/.cache/mozilla/firefox/...frequencyCap.json.tmp |  3,030 |

#### theia_e5
| Path                                             | Count |
| ------------------------------------------------ | ----: |
| /run/motd.new                                    | 4,944 |
| /run/ConsoleKit/database~                        | 1,903 |
| /home/admin/.mozilla/firefox/.../recovery.js.tmp |   232 |
| /home/admin/eraseme                              |   223 |

```sql
SELECT path, COUNT(*) as cnt FROM file_node_table 
WHERE path IS NOT NULL AND path != '' 
GROUP BY path ORDER BY cnt DESC LIMIT 10;
```

---

## 3. Statistics of 'Netflow Objects' (`netflow_node_table`) in all databases

### 3.1 Total Count & Address Diversity

| Database      | Netflow Nodes | Unique src_addr | Unique dst_addr |     Local |  External |
| ------------- | ------------: | --------------: | --------------: | --------: | --------: |
| cadets_e3     |       155,322 |               4 |           1,170 |     5,732 |   149,590 |
| cadets_e5     |       219,384 |               7 |              43 |     9,228 |   210,156 |
| clearscope_e3 |       279,694 |           3,778 |               1 |         0 |   279,694 |
| clearscope_e5 |       104,022 |               9 |               1 |         0 |   104,022 |
| optc_051      |     4,184,346 |           7,361 |             174 | 1,349,787 | 2,834,559 |
| optc_201      |     4,696,258 |           7,391 |             412 | 1,492,429 | 3,203,829 |
| optc_501      |     3,547,821 |           7,394 |           1,167 | 1,129,194 | 2,418,627 |
| theia_e3      |       186,100 |               9 |             202 |        37 |   186,063 |
| theia_e5      |        36,747 |               3 |             151 |        75 |    36,672 |

Observations:
- **clearscope** netflow is entirely external (`0.0.0.0`, `::`, IPv6 mapped addresses).
- **optc** has the highest netflow volumes (4M+ connections) and diverse source addresses.
- **cadets** and **theia** have very few source IPs but many destinations.
- **optc** has significant local traffic (~30%), others are mostly external.

```sql
SELECT COUNT(*) FROM netflow_node_table;
SELECT COUNT(DISTINCT src_addr), COUNT(DISTINCT dst_addr) FROM netflow_node_table;
SELECT CASE WHEN src_addr LIKE '10.%' OR src_addr LIKE '127.%' OR src_addr LIKE '192.168.%'
  THEN 'local' ELSE 'external' END as type, COUNT(*) as cnt
FROM netflow_node_table GROUP BY type;
```

### 3.2 Top src_addr per Database (Top 3)

| Database      | src_addr      |   Count |
| ------------- | ------------- | ------: |
| cadets_e3     | 128.55.12.73  |  91,453 |
|               | localhost     |  58,137 |
|               | 127.0.0.1     |   5,700 |
| cadets_e5     | 128.55.12.106 |  81,166 |
|               | 128.55.12.51  |  72,657 |
|               | 128.55.12.75  |  56,333 |
| clearscope_e3 | 0.0.0.0       |  52,415 |
|               | 128.55.12.166 |  48,412 |
|               | ::            |  42,652 |
| optc_051      | 142.20.56.52  |  93,380 |
|               | 127.0.0.1     |  81,368 |
|               | 142.20.61.130 |  27,596 |
| optc_201      | 142.20.56.202 | 170,744 |
|               | 127.0.0.1     | 114,962 |
|               | 165.101.35.5  |  46,265 |
| theia_e3      | 128.55.12.110 | 185,142 |
|               | NETLINK       |     915 |
|               | 10.0.6.60     |      29 |
| theia_e5      | 128.55.12.110 |  36,672 |
|               | 127.0.0.1     |      52 |
|               | 10.0.6.60     |      23 |

```sql
SELECT src_addr, COUNT(*) as cnt FROM netflow_node_table 
GROUP BY src_addr ORDER BY cnt DESC LIMIT 5;
```

### 3.3 Top dst_addr per Database (Top 3)

| Database      | dst_addr          |     Count |
| ------------- | ----------------- | --------: |
| cadets_e3     | 128.55.12.10      |   115,685 |
|               | 128.55.12.166     |    17,696 |
|               | 127.0.0.1         |     5,848 |
| cadets_e5     | 128.55.12.10      |    97,540 |
|               | 128.55.12.122     |    71,857 |
|               | 162.99.3.50       |    14,513 |
| clearscope_e3 | 0                 |   279,694 |
| clearscope_e5 | (empty)           |    56,222 |
|               | 0                 |    47,800 |
| optc_051      | 224.0.0.252       | 1,919,187 |
|               | ff02::1:3         | 1,917,465 |
|               | 142.20.56.52      |    92,278 |
| optc_201      | 224.0.0.252       | 2,079,185 |
|               | ff02::1:3         | 2,076,634 |
|               | 142.20.56.202     |   166,771 |
| theia_e3      | NA / 128.55.12.10 |   ~28,200 |
|               | 64.191.208.114    |     6,619 |
|               | 67.28.122.168     |     6,106 |
| theia_e5      | 128.55.12.10      |    27,644 |
|               | 128.55.12.122     |     2,870 |
|               | 162.99.3.50       |     1,493 |

```sql
SELECT dst_addr, COUNT(*) as cnt FROM netflow_node_table 
GROUP BY dst_addr ORDER BY cnt DESC LIMIT 5;
```

---

## 4. Statistics of 'Subject Objects' (`subject_node_table`) in all databases

### 4.1 Total Subject Count & Command Diversity

| Database      | Subject Nodes | Unique cmds | Null Path |
| ------------- | ------------: | ----------: | --------: |
| cadets_e3     |       224,146 |         130 |   224,146 |
| cadets_e5     |     6,576,457 |         126 | 6,576,457 |
| clearscope_e3 |        12,469 |          44 |    12,469 |
| clearscope_e5 |        73,097 |         109 |    73,097 |
| optc_051      |        50,626 |       1,132 |        33 |
| optc_201      |        99,693 |       1,374 |        29 |
| optc_501      |        70,913 |       1,171 |        81 |
| theia_e3      |       278,363 |       1,569 |         0 |
| theia_e5      |     1,264,440 |       1,566 |         0 |

Observations:
- **cadets** and **clearscope** have all null paths — no file path recorded for subjects.
- **theia** has full path information (0 null paths).
- **optc** has the highest command diversity (1,100+) and the most detailed command lines.

```sql
SELECT COUNT(*), COUNT(DISTINCT cmd), 
       COUNT(*) FILTER (WHERE path IS NULL OR path = '') as null_path
FROM subject_node_table;
```

### 4.2 Top Commands per Database (Top 5)

#### cadets_e3
| Command |  Count |
| ------- | -----: |
| sleep   | 61,840 |
| vmstat  | 30,877 |
| top     | 15,618 |
| date    | 15,469 |
| lsof    | 15,452 |

#### cadets_e5
| Command |     Count |
| ------- | --------: |
| cc      | 2,985,748 |
| sh      | 2,159,141 |
| sed     |   591,201 |
| sshd    |   318,807 |
| bash    |    80,012 |

#### clearscope_e3
| Command                       | Count |
| ----------------------------- | ----: |
| android.process.acore         | 3,507 |
| com.android.inputmethod.latin | 3,001 |
| com.android.smspush           | 1,538 |
| com.android.quicksearchbox    | 1,025 |
| com.android.launcher3         |   829 |

#### clearscope_e5
| Command                                   |  Count |
| ----------------------------------------- | -----: |
| /system/bin/app_process64                 | 54,158 |
| /system/bin/ping                          | 16,643 |
| /system/bin/dex2oat                       |    490 |
| com.android.internal.os.WebViewZygoteInit |    301 |
| /system/bin/crash_dump64                  |    173 |

#### optc_051
| Command                                                           |  Count |
| ----------------------------------------------------------------- | -----: |
| (empty)                                                           | 26,560 |
| /??/C:/Windows/system32/conhost.exe 0xffffffff -ForceV1           |  6,164 |
| C:/Windows/SYSTEM32/cmd.exe /c "C:/ncr/DeleteArchiveSecurity.bat" |  4,503 |
| ping -w 2000 -n 2 127.0.0.1                                       |  2,062 |
| C:/Windows/System32/svchost.exe -k wsappx                         |    610 |

#### optc_201
| Command                                                           |  Count |
| ----------------------------------------------------------------- | -----: |
| (empty)                                                           | 55,842 |
| /??/C:/Windows/system32/conhost.exe 0xffffffff -ForceV1           |  6,927 |
| C:/Windows/SYSTEM32/cmd.exe /c "C:/ncr/DeleteArchiveSecurity.bat" |  5,152 |
| netstat -n                                                        |  3,338 |
| ping -w 2000 -n 2 127.0.0.1                                       |  3,183 |

#### optc_501
| Command                                                           |  Count |
| ----------------------------------------------------------------- | -----: |
| (empty)                                                           | 37,704 |
| /??/C:/Windows/system32/conhost.exe 0xffffffff -ForceV1           |  7,226 |
| C:/Windows/SYSTEM32/cmd.exe /c "C:/ncr/DeleteArchiveSecurity.bat" |  4,940 |
| ping -w 2000 -n 2 127.0.0.1                                       |  2,024 |
| netstat -n                                                        |  1,920 |

#### theia_e3
| Command                                                           |  Count |
| ----------------------------------------------------------------- | -----: |
| /usr/bin/firefox                                                  | 77,985 |
| -bash                                                             | 53,060 |
| /bin/sh -e /usr/lib/update-notifier/update-motd-updates-available | 25,917 |
| /usr/lib/postgresql/9.1/bin/postgres -D /var/lib/...              | 10,800 |
| /usr/bin/python /usr/lib/unity-scope-video-remote                 |  7,003 |

#### theia_e5
| Command                                                           |   Count |
| ----------------------------------------------------------------- | ------: |
| /bin/sh -e /usr/lib/update-notifier/update-motd-updates-available | 526,317 |
| -bash                                                             | 372,550 |
| run-parts --lsbsysinit /etc/update-motd.d                         |  45,552 |
| /bin/sh -e /usr/lib/update-notifier/update-motd-hwe-eol           |  35,428 |
| /bin/sh /etc/update-motd.d/50-landscape-sysinfo                   |  33,740 |

```sql
SELECT cmd, COUNT(*) as cnt FROM subject_node_table 
GROUP BY cmd ORDER BY cnt DESC LIMIT 10;
```

---

## Key Insights Summary

1. **Scale**: The largest dataset is cadets_e5 (1.07B events), followed by clearscope_e5 (199M) and theia_e5 (141M). The E3 (2018) datasets are much smaller than E5 (2019) ones.

2. **Operation Dominance**: READ and WRITE dominate across all datasets (>80% of events). EXECUTE/FORK/EXIT (process lifecycle) events are a small fraction (<3%).

3. **Network Traffic**: optc datasets have the richest netflow data (4M+ connections) with diverse IPs. cadets and theia use a small set of internal IPs (mostly 128.55.12.x). clearscope netflow is primarily `0.0.0.0`/`::` (raw socket monitoring).

4. **File Access**: optc shows file content deduplication (hash_id reuse), while cadets/theia have one-to-one file-to-hash mapping. The most accessed files in cadets are mail locks and temp files; in optc they're NCR ATM-related files; in theia they're browser and PostgreSQL temp files.

5. **Process Activity**: cadets_e5 shows heavy compilation activity (cc, sh, sed — build server). theia shows heavy web browser and update-notifier activity. optc shows Windows ATM system processes. clearscope is Android app processes.

6. **Temporal Patterns**: E3 datasets span ~April 2–13, 2018; E5 datasets span ~May 7–20, 2019; optc spans ~Sept 17–25, 2019. Activity is mostly continuous with some daily variation.


# Ground Truth (Maliciuos)

```bash
# prompt
Great, we have the statistics for all 9 databases. Here i have the folder of ground truth data (malicious nodes or events), so I want you to do some statistics on them for each database, so we can see later of if we can get some insights to help in detecting malicious activities in the future. (folder: /srv/lustre01/project/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/xp/PIDSMaker/Ground_Truth/orthrus) use only the databases we already worked on (cadets_e3, cadets_e5, clearscope_e3, clearscope_e5, optc_051, optc_201, optc_501, theia_e3, theia_e5).
```
---

## 5. Ground Truth Analysis — Malicious Nodes & Events

> Ground truth source: `/xp/PIDSMaker/Ground_Truth/orthrus/`
> Mapping: `E3-CADETS` → `cadets_e3`, `E5-CADETS` → `cadets_e5`, `E3-CLEARSCOPE` → `clearscope_e3`,
> `E5-CLEARSCOPE` → `clearscope_e5`, `E3-THEIA` → `theia_e3`, `E5-THEIA` → `theia_e5`,
> `h051` → `optc_051`, `h201` → `optc_201`, `h501` → `optc_501`

### 5.1 Ground Truth Overview

| Database      | GT Entries | Subjects | Files | Netflows | Attacks |
| ------------- | ---------: | -------: | ----: | -------: | ------: |
| cadets_e3     |         79 |       41 |    19 |       19 |       4 |
| cadets_e5     |        126 |      105 |    11 |       10 |       2 |
| clearscope_e3 |         41 |        1 |    40 |        0 |       1 |
| clearscope_e5 |         52 |       31 |    21 |        0 |       4 |
| theia_e3      |        119 |       96 |    11 |       11 |       2 |
| theia_e5      |         70 |       64 |     2 |        4 |       1 |
| optc_051      |        114 |       59 |     6 |       49 |       1 |
| optc_201      |      2,905 |       42 |     5 |    2,858 |       1 |
| optc_501      |        749 |       23 |    26 |      700 |       1 |

```sql
SELECT entity_type, COUNT(DISTINCT node_uuid) 
FROM gt_staging GROUP BY entity_type;
```

### 5.2 Malicious Events per Database

| Database      | Total DB Events | Malicious (src) | Malicious (dst) | Total Malicious | % of DB |
| ------------- | --------------: | --------------: | --------------: | --------------: | ------: |
| cadets_e3     |      36,484,667 |       2,683,955 |         265,753 |   **2,943,406** |   8.07% |
| cadets_e5     |   1,075,370,668 |               — |               — |               — |       — |
| clearscope_e3 |      18,316,129 |          13,208 |          16,052 |      **27,931** |   0.15% |
| clearscope_e5 |     198,794,211 |      31,871,075 |      32,478,489 |  **54,250,986** |  27.29% |
| theia_e3      |      44,366,117 |      13,150,864 |       2,037,807 |               — |       — |
| theia_e5      |     140,994,662 |          15,723 |       1,269,915 |               — |       — |
| optc_051      |      31,611,029 |          65,635 |         591,588 |     **657,223** |   2.08% |
| optc_201      |      38,151,059 |       1,260,401 |      11,122,146 |  **12,382,547** |  32.46% |
| optc_501      |      28,049,214 |         376,657 |         731,666 |   **1,108,323** |   3.95% |

Note: `as_dst` counts include events where a malicious node is the destination. Some events overlap (both src and dst are malicious), so `Total` may differ from `src + dst`.

```sql
WITH malicious_nodes AS (
    SELECT s.hash_id, s.index_id::text as index_id
    FROM gt_staging g
    JOIN subject_node_table s ON g.node_uuid = s.node_uuid AND g.entity_type = 'subject'
    UNION ALL ...
)
SELECT COUNT(*) FROM event_table e WHERE ...
```

### 5.3 Malicious Operations Distribution

#### cadets_e3
| Operation           |     Count |
| ------------------- | --------: |
| EVENT_READ          | 1,856,113 |
| EVENT_CLOSE         |   616,375 |
| EVENT_OPEN          |   269,590 |
| EVENT_CREATE_OBJECT |   176,050 |
| EVENT_WRITE         |    19,236 |
| EVENT_RECVFROM      |     2,309 |
| EVENT_LSEEK         |     1,741 |
| EVENT_MMAP          |       548 |
| EVENT_ACCEPT        |       456 |

#### clearscope_e3
| Operation      |  Count |
| -------------- | -----: |
| EVENT_READ     | 10,497 |
| EVENT_WRITE    |  6,994 |
| EVENT_OPEN     |  5,821 |
| EVENT_RECVFROM |  1,882 |
| EVENT_CONNECT  |  1,066 |
| EVENT_SENDTO   |    978 |
| EVENT_SENDMSG  |    693 |

#### optc_051
| Operation |   Count |
| --------- | ------: |
| START     | 359,334 |
| MESSAGE   | 111,048 |
| WRITE     |  81,560 |
| READ      |  51,197 |
| MODIFY    |  35,578 |
| OPEN      |  10,198 |
| RENAME    |   2,479 |
| CREATE    |     833 |
| DELETE    |     424 |
| TERMINATE |      43 |

#### optc_201
| Operation |   Count |
| --------- | ------: |
| MODIFY    | 546,971 |
| OPEN      | 389,251 |
| WRITE     | 246,723 |
| READ      |  31,492 |
| START     |  25,542 |
| MESSAGE   |  16,811 |
| RENAME    |     322 |
| DELETE    |     275 |
| TERMINATE |      35 |

#### optc_501
| Operation |   Count |
| --------- | ------: |
| MODIFY    | 181,180 |
| OPEN      | 105,083 |
| WRITE     |  64,223 |
| MESSAGE   |   7,112 |
| DELETE    |   6,680 |
| START     |   2,251 |
| READ      |   1,148 |
| RENAME    |     662 |
| TERMINATE |      17 |

```sql
SELECT e.operation, COUNT(DISTINCT e._id) as cnt
FROM event_table e
JOIN _mn m ON (e.src_node = m.hash_id AND e.src_index_id = m.index_id)
           OR (e.dst_node = m.hash_id AND e.dst_index_id = m.index_id)
GROUP BY e.operation ORDER BY cnt DESC;
```

### 5.4 Malicious Events by Attack

| Database      | Attack                     | Malicious Events |
| ------------- | -------------------------- | ---------------: |
| cadets_e3     | Nginx_Backdoor_12          |        2,921,269 |
|               | Nginx_Backdoor_13          |          650,013 |
|               | Nginx_Backdoor_11          |            2,507 |
|               | Nginx_Backdoor_06          |            1,957 |
| clearscope_e3 | clearscope_e3_firefox_0411 |           27,931 |
| optc_051      | h051_0925                  |          657,223 |
| optc_201      | h201_0923                  |       12,382,547 |
| optc_501      | h501_0924                  |        1,108,323 |

### 5.5 Malicious Commands (Subjects)

#### cadets_e3
| Command   | GT Nodes | Attacks            |
| --------- | -------: | ------------------ |
| sshd      |        9 | Nginx_Backdoor_13  |
| sendmail  |        8 | Nginx_Backdoor_12  |
| test      |        4 | Nginx_Backdoor_12  |
| python2.7 |        3 | Nginx_Backdoor_12  |
| minions   |        2 | Nginx_Backdoor_12  |
| nginx     |    1 per | _06, _11, _12, _13 |
| vUgefal   |        1 | Nginx_Backdoor_06  |
| main      |        1 | Nginx_Backdoor_12  |
| wget      |        1 | Nginx_Backdoor_13  |
| pEja72mA  |        1 | Nginx_Backdoor_13  |

#### cadets_e5
| Command | GT Nodes | Attacks             |
| ------- | -------: | ------------------- |
| main    |       56 | Nginx_Drakon_APT_17 |
| cat     |       28 | Nginx_Drakon_APT_17 |
| bash    |        4 | Nginx_Drakon_APT_17 |
| nginx   |        4 | Nginx_Drakon_APT    |
| sshd    |        3 | Nginx_Drakon_APT_17 |

#### theia_e3
| Command             | GT Nodes | Attack                            |
| ------------------- | -------: | --------------------------------- |
| ./gtcache           |       34 | Browser_Extension_Drakon_Dropper  |
| /usr/bin/firefox    |       20 | Firefox_Backdoor_Drakon_In_Memory |
| /home/admin/clean   |       19 | Firefox_Backdoor_Drakon_In_Memory |
| /usr/bin/firefox    |       10 | Browser_Extension_Drakon_Dropper  |
| /home/admin/profile |        6 | Firefox_Backdoor_Drakon_In_Memory |

#### theia_e5
| Command            | GT Nodes | Attack                                           |
| ------------------ | -------: | ------------------------------------------------ |
| sshd: admin [priv] |       19 | THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject |
| firefox            |       12 | ...                                              |
| sshd: [accepted]   |       11 | ...                                              |
| /usr/sbin/sshd -D  |       11 | ...                                              |

#### optc_051
| Command                                                        | GT Nodes |
| -------------------------------------------------------------- | -------: |
| (empty)                                                        |       30 |
| /??/C:/Windows/system32/conhost.exe 0xffffffff -ForceV1        |        1 |
| "C:/Users/dcoombes/AppData/Local/Temp/cKfGW.exe"               |        1 |
| "C:/Users/dcoombes/Downloads/update.exe"                       |        1 |
| C:/Windows/Temp/rad*.tmp/biGuWCmNsuCIG.exe (multiple variants) |       22 |

#### optc_201
| Command                                                              | GT Nodes |
| -------------------------------------------------------------------- | -------: |
| C:/Windows/SysNative/qwinsta.exe zleazer /server 142.20.61.187       |       15 |
| (empty)                                                              |       11 |
| C:/Windows/system32/cmd.exe /c "C:/Windows/SysNative/qwinsta.exe..." |        8 |
| /??/C:/Windows/system32/conhost.exe 0xffffffff -ForceV1              |        2 |
| cmdkey.exe /generic:142.20.61.187 /user:systemia.com/zleazer /pass.. |        1 |

#### optc_501
| Command                                                              | GT Nodes |
| -------------------------------------------------------------------- | -------: |
| (empty)                                                              |       12 |
| "C:/Program Files (x86)/Microsoft Office/Office15/WINWORD.EXE" ...   |        2 |
| "C:/Windows/system32/plink.exe" sports.com -R 4000:127.0.0.1:3389... |        1 |
| "C:/Users/sysadmin/Downloads/fileTransfer1000.exe" —help             |        1 |
| powershell -NoProfile -Sta -WindowStyle 1 -EncodedCommand ...        |        1 |

```sql
SELECT s.cmd, COUNT(DISTINCT g.node_uuid) as gt_nodes, g.attack_name
FROM gt_staging g
JOIN subject_node_table s ON g.node_uuid = s.node_uuid
WHERE g.entity_type = 'subject'
GROUP BY s.cmd, g.attack_name ORDER BY COUNT(*) DESC;
```

### 5.6 Malicious File Paths

#### cadets_e3 (Nginx Backdoor)
| Path                                                                    | Attack            |
| ----------------------------------------------------------------------- | ----------------- |
| /tmp/vUgefal                                                            | Nginx_Backdoor_06 |
| /tmp/main, /tmp/minions, /tmp/font, /tmp/XIM, /tmp/test, /tmp/tmux-1002 | _12               |
| /var/cache/salt/minion/proc/...                                         | _12               |
| /var/log/netlog, /var/log/salt/minion, /var/log/sendmail                | _12               |
| /tmp/pEja72mA                                                           | Nginx_Backdoor_13 |
| /usr/home/user/eraseme                                                  | Nginx_Backdoor_13 |

#### clearscope_e3 (Firefox)
| Path                                                         | Attack                     |
| ------------------------------------------------------------ | -------------------------- |
| /data/data/org.mozilla.fennec_firefox_dev/cache/* (40 files) | clearscope_e3_firefox_0411 |

#### theia_e3 (Drakon)
| Path                                         | Attack                            |
| -------------------------------------------- | --------------------------------- |
| /etc/firefox/native-messaging-hosts/gtcache  | Browser_Extension_Drakon_Dropper  |
| /etc/firefox/native-messaging-hosts/pass_mgr | ...                               |
| /tmp/memtrace.so                             | ...                               |
| /var/log/wdev, /var/log/xdev                 | ...                               |
| /home/admin/clean, /home/admin/profile       | Firefox_Backdoor_Drakon_In_Memory |
| /dev/glx_alsa_675                            | ...                               |

#### theia_e5 (SSH + Drakon)
| Path             | Attack                                           |
| ---------------- | ------------------------------------------------ |
| /run/shm/lc      | THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject |
| /var/log/sshdlog | ...                                              |

#### optc_051 (Ransomware/Backdoor)
| Path                                                        |
| ----------------------------------------------------------- |
| /Device/HarddiskVolume1/$Extend/$UsnJrnl:$J                 |
| /Device/HarddiskVolume1/Python27/Lib/site-packages/...      |
| /Device/HarddiskVolume1/Users/dcoombes/Downloads/update.exe |

#### optc_201 (Credential Theft)
| Path                                                      |
| --------------------------------------------------------- |
| /Device/HarddiskVolume1/$Mft                              |
| /Device/HarddiskVolume1/Users/zleazer/Downloads/runme.bat |

#### optc_501 (Data Exfiltration)
| Path                                               |
| -------------------------------------------------- |
| /Device/HarddiskVolume1/$Mft                       |
| /Device/HarddiskVolume1/.../payroll.docx           |
| /Device/HarddiskVolume1/.../export.zip             |
| /Device/HarddiskVolume1/.../fileTransfer1000.exe   |
| /Device/HarddiskVolume1/windows/system32/plink.exe |

```sql
SELECT f.path, g.attack_name FROM gt_staging g
JOIN file_node_table f ON g.node_uuid = f.node_uuid
WHERE g.entity_type = 'file' AND f.path IS NOT NULL AND f.path != ''
ORDER BY g.attack_name, f.path;
```

### 5.7 Malicious Netflow Analysis

| Database  | Attacker IP        | Port  | Target IP      | Port  | Attack                            |
| --------- | ------------------ | ----- | -------------- | ----- | --------------------------------- |
| cadets_e3 | 128.55.12.73       | 80    | 81.49.200.166  | 43530 | Nginx_Backdoor_06                 |
|           | 128.55.12.73       | 80    | 25.159.96.207  | 45203 | Nginx_Backdoor_12                 |
|           | 128.55.12.73       | 46517 | 76.56.184.25   | 80    | Nginx_Backdoor_12                 |
| cadets_e5 | 128.55.12.51       | 80    | 128.55.12.167  | 58356 | Nginx_Drakon_APT                  |
|           | 128.55.12.51       | 40039 | 98.23.182.25   | 80    | Nginx_Drakon_APT                  |
| theia_e3  | 128.55.12.110      | 33100 | 141.43.176.203 | 80    | Firefox_Backdoor_Drakon_In_Memory |
|           | 128.55.12.110      | 45226 | 161.116.88.72  | 80    | ...                               |
| theia_e5  | 128.55.12.110      | 22    | 128.55.12.167  | 58983 | THEIA_1_Drakon_APT                |
|           | 128.55.12.110      | 40837 | 208.203.20.42  | 80    | ...                               |
| optc_051  | 142.20.56.52       | 49842 | 53.192.68.50   | 8080  | h051_0925 (C2)                    |
|           | 142.20.56.52       | 3389  | 53.192.68.50   | 58878 | h051_0925 (RDP)                   |
| optc_201  | **132.197.158.98** | 80    | 142.20.56.202  | *     | h201_0923 (C2 → victim)           |
| optc_501  | 142.20.57.246      | 50055 | 202.6.172.98   | 443   | h501_0924 (HTTPS C2)              |
|           | 142.20.57.246      | 3389  | 53.192.68.50   | 58878 | h501_0924 (RDP tunnel)            |

Patterns observed:
- **CADETS**: Nginx backdoor initiates outbound HTTP (port 80) connections to external IPs from `128.55.12.73`
- **Theia**: Firefox-based Drakon implant connects to C2 on port 80; SSH backdoor on port 22
- **OPTC**: RDP (3389), HTTP proxy (8080), HTTPS (443) C2 channels; `53.192.68.50` is a common C2 across optc
- **OPTC 201**: Traffic is *inbound* from `132.197.158.98:80` to victim, suggesting reverse shell

```sql
SELECT SPLIT_PART(g.entity_detail, ':', 1) as src_addr,
       SPLIT_PART(SPLIT_PART(g.entity_detail, '->', 2), ':', 1) as dst_addr,
       g.attack_name
FROM gt_staging g WHERE g.entity_type = 'netflow';
```

### 5.8 Temporal Distribution of Malicious Events

#### cadets_e3
| Day        | Malicious Events |
| ---------- | ---------------: |
| 2018-04-02 |            8,289 |
| 2018-04-03 |          254,654 |
| 2018-04-04 |          212,612 |
| 2018-04-05 |          213,982 |
| 2018-04-06 |          202,375 |
| 2018-04-07 |          214,647 |
| 2018-04-08 |          218,460 |
| 2018-04-09 |          214,996 |
| 2018-04-10 |          211,683 |
| 2018-04-11 |          233,888 |
| 2018-04-12 |      **490,809** |
| 2018-04-13 |      **467,011** |

#### clearscope_e3
| Day        | Malicious Events |
| ---------- | ---------------: |
| 2018-04-07 |              145 |
| 2018-04-08 |              432 |
| 2018-04-11 |       **26,758** |
| 2018-04-13 |              596 |

#### optc_051
| Day        | Malicious Events |
| ---------- | ---------------: |
| 2019-09-17 |            1,659 |
| 2019-09-18 |            2,509 |
| 2019-09-19 |            5,449 |
| 2019-09-20 |            3,330 |
| 2019-09-21 |           12,884 |
| 2019-09-22 |            5,002 |
| 2019-09-23 |            5,506 |
| 2019-09-24 |            5,462 |
| 2019-09-25 |       **23,834** |

#### optc_201
| Day        | Malicious Events |
| ---------- | ---------------: |
| 2019-09-17 |           34,340 |
| 2019-09-18 |           25,210 |
| 2019-09-19 |           33,561 |
| 2019-09-20 |      **213,391** |
| 2019-09-21 |    **3,303,292** |
| 2019-09-22 |    **3,326,548** |
| 2019-09-23 |    **3,331,294** |
| 2019-09-24 |        1,971,355 |
| 2019-09-25 |           14,497 |

#### optc_501
| Day        | Malicious Events |
| ---------- | ---------------: |
| 2019-09-19 |            8,907 |
| 2019-09-20 |              611 |
| 2019-09-21 |           10,354 |
| 2019-09-22 |           10,628 |
| 2019-09-23 |           10,650 |
| 2019-09-24 |      **487,608** |
| 2019-09-25 |      **537,428** |

```sql
SELECT DATE(TO_TIMESTAMP(e.timestamp_rec / 1000000000)) as day, COUNT(*) as cnt
FROM event_table e JOIN _mn m ON (...) GROUP BY day ORDER BY day;
```

### 5.9 Key Ground Truth Insights

1. **Attack volume varies dramatically** — clearscope_e5 has 27% events malicious, clearscope_e3 only 0.15%.

2. **READ dominates malicious activity** in CADETS/ClearScope attackers (`nginx backdoor` reads files). In OPTC, START/MODIFY/WRITE dominate (process creation + file modification).

3. **C2 infrastructure patterns**:
   - `53.192.68.50` appears across optc_051 as multi-protocol C2 (ports 8080, 8000, 58878/3389 RDP)
   - `128.55.12.73` (CADETS) and `128.55.12.110` (Theia) are compromised internal hosts making outbound connections
   - `132.197.158.98` is an external C2 connecting **inbound** to optc_201 victim

4. **Malicious commands are often empty** (OPTC) or common Linux commands used nefariously (cat, main, bash, sshd, nginx). The diversity is low compared to benign commands.

5. **Temporal clustering**: Malicious activity spikes around attack dates (e.g., Apr 12-13 for E3, Sep 20-24 for optc_201, Sep 19-25 for optc_501).

6. **Linked vs unlinked nodes**: CADETS/ClearScope/Theia link events via hash_id, while OPTC links directly via node_uuid — an important implementation detail for detection systems.
