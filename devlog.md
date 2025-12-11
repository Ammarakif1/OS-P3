# 2025-12-06 11pm

spending today going over P3. The project looking like a CLI tool that uses B-Tree 512 byte blocks with cap of three nodes in memory comands are something like creation as well as loading of index files inserting searching of keys, printing contents manipulation of CSV data. the plan is to use python for it!


# 2025-11-07 10pm

mapped out the on-disk layout:  512 byte   storing  string, root pointer, and next block id. fixed-size node blocks holding up to 19 key/value pairs and 20 child pointers. 

# 2025-11-08 12pm

implemented the helpers serialization block allocation a context-managed NodeManager that enforces the three-node ceiling. Filled in the B-Tree core logic

# 2025-11-09 1pm

finished up the CLI commands create/insert/search/load/print/extract 
fixed some bugs