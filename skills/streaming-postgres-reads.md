# Skill: Stream Postgres reads with a server-side cursor

`psycopg2` cursors created via `connect.cursor()` (the default) buffer the
**entire result set client-side** after `cur.execute(...)`. For E5-scale node
and event tables this can buffer 10–50GB before the first row is processed.

Use `pidsmaker.utils.utils.stream_query(connect, sql, name=..., itersize=10000)`
to drive a *named* (server-side) cursor instead. Rows are fetched from the
server in batches of `itersize` and yielded one-by-one to the caller.

## Quick example
```python
from pidsmaker.utils.utils import init_database_connection, stream_query

cur, connect = init_database_connection(cfg)
for row in stream_query(connect, "select * from netflow_node_table;", name="my_stream"):
    ...
```

## Gotchas
- Each named cursor needs a unique `name=` if multiple streams overlap on the
  same connection.
- Named cursors are read-only and cannot be used for `INSERT`/`UPDATE`.
- Wrapping with `for row in ...` consumes the generator; you cannot rewind.
- `stream_query` closes the cursor in a `finally`, so partial iteration is safe.
