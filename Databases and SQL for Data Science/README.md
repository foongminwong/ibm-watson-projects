# Databases and SQL for Data Science

## Introduction to SQL for Data Science

**Basic SQL**
* SQL (Structured English Query Language) - a language used for relational database to query/get data out of a database
* Data - a collection of facts in the form of words, numbers or pictures, critical assets for business and collected practically (e.g.: address, phone #, account #, etc.)
* Database - a repository of data/a program that stores data which also provides the functionality for adding, modifying and querying data
* Relational database - data stored in tabular form (organzied in tables likes spreadsheet), can form relationaships between tables
* Database management system/DBMS - a set of software tools for data in database
* Relational database management system/RDBMS - a set of tools that controls the data such as access, organziation and storage (e.g: mySQL, Oracle Database, Db2 Warehouse on Cloud, DB2 Express-C)
* Basic SQL commands: CREATE, INSERT, SELECT, UPDATE, DELETE

**Create a Database Instance on Cloud (a service isntance on an IBM Db2 on cloud database)**
* cloud database - database service built and accessed through a cloud platform (users install software on a cloud infrastructure to implement the database)
* Pros: 
    * Can access cloud databases from virtually anywhere using a vendors API or web interface 
    * Scalability - Can expand their storage capacities during runtimeto accommodate changing needs
    * Disaster recovery - kept data secure through backups on remote servers
    * e.g.: IBM DB2 on Cloud, COmpose for PostgreSQL, Oracle Database Cloud, Microsoft Azure Cloud, SQL Databse, Amazon Relational Database Services 

**CREATE TABLE Statement**
* DDL (Data Definition Language) statements - define, change or drop data (e.g. CREATE)
* DML (Data Manipulation Language) statements: read & modify data (e.g. SELECT, INSERT, UPDATE)
* Columns -> Entity attributes (e.g. char - character string of a fixed length, varcahr - character string of a variable length)
* Priamry key - no duplicate values can exist, uniquely identifies each tuple or row in a table, constraint prevents duplicate values in table
* Constraint NOT NULL - ensures fields cannot contain a null value

**SELECT Statement**
* Retrieval of data
* SELECT statement = query, output that we get from executing query - result set/result table
* WHERE Clause - to restrict the Result Set
* WHERE clause requires a predicate
* Predicate - is conditioned evaluates to true, false or unknown, used in search condition

**COUNT, DISTINCT, LIMIT (expressions used with SELECT statements)**

**INSERT Statement**
* Populate with /insert data
* A single INSERT statement can be used to inset 1 or multiple rows in a table

**UPDATE and DELETE Statements**

## Relational Database Concepts

**Informationa and Data Models**
* Information model - an abstract formal representation of entities (can be from real world) that includes their properties, relationships, operations that can be performed on them, at conceptual levels & defines relationships between objects (e.g. Hierarchical - organziation chart ina tree structure)
* Data model - concrete level, specific & include details, a blueprint of any database system
(e.g. Relational Model - most used, allows data independence such as logical data independence, physical data independence, physical storage independence) (e.g. entity-relationship/ER data model - alternative to a relational data model)
* Entity Relationship Diagram (ERD) - represents entities called tables and their relationships (ER Diagram = Entities [rectangle] + Attributes [ovals])
* Entity-Relationship Model - a database as a collection of entities (objects that exist independently of any other entities in the database), to design relational databases

**Types of Relationships**
* Relationship = entities sets [rectangle] + relationship sets [diamond with lines connecting associated entities]+ crows foot notations
* 1-to-1 (entity || diamond || entity)
* 1-to-many OR many-to-1 (entity || diamond < Author) 
* many-to-many (book > authored by < author)

**Mapping Entities to Tables**
* Entity (Book) = Table, Attributes (Title, Edition, Description, ISBN, etc.) = Columns

**Relational Model Concepts**
* Relation Model = Relation + Sets
* Set - unordered collection of distinct elements, collection of items of the same type, no order & no duplicates
* Relational Databse = set of relations
* Relation - mathematical term for a table (a combination of rows and columns)
* Relation:
    * Relational Schema - specify name of a relation and attributes (e.g. `AUTHOR(Author_ID:char, lastname:varchar, firstname: varchar, country: char)`)
    * Relational Instance - a table made up of attributes or columns
        * Columns = attributes = field
        * Row = Tuple
* Relation: Degree & Cardinality
    * Degree = the number of attributes in a relation
    * Cardinality = the number of tuples or rows
* Summary:
    * Relational Model: based on the concept of relation
    * Relation:
        * A mathematical concept-based, matheamtical term for table
        * Made up of 2 parts: Relational schema and relational Instance
    * Relational schema: specifies relation name and attributes
    * Relation Instance: a table made up of attributes and tuples
    * Degree: the numebr of attributes
    * Cardinality: the number of tuples

**Additional Information**
* Create Schema = A SQL schema is identified by a schema name, and includes a authorization identifier to indicate the user or account who owns the schema. Schema elements include tables, constraints, views, domains and other constructs that describe the schema. (e.g. LIBRARY schema has multiple tables such as AUTHOR, BOOK)
* Database -> Schema -> Tables

## Advanced SQL

**Relational Model Constraints**
* Primary key (PK) - uniquely identifies each row in table
* Foreign key (FK) - set of columns referring to a PK of another table
* Parent table = a table containing PK that is realted to at least 1 FK
* Dependent table = a table containing 1 or more FK

**Relational Model Constraints - Advanced**
* 6 constraints:
    * Entity Integrity Constraint - no attribute participating in the PK of a relation is allowed to accept null values (PK cannot have an unknown value!) (Reason: A PK uniquely identifies each row in a table. If PK could have NULL values, you could end up with duplicate rows in a table)
    * Referential Integrity Constraint - ensures the validity of the data using a combination of PK and FK
    * Semantic Integrity Constraint - refers to correctness of the meaning of the data
    * Domain Constraint - specifies the permissible values for a given attribute
    * Null Constraint - specifies that attribute values cannot be null
    * Check Constraint - enforces domain integrity by limiting the values that are accepted by an attribute



## Accessing databases using Python
**How to Access Databases Using Python**
* DB stands for database
* create an instance in the cloud, connect to a DB, query data from the DB using SQL, analyze data using Python, describe SQL API, proprietary API used by popular SQL-based DBM systems
* Python - popular scripting language for connecting to DBs (packages: NumPy, pandas, matplotlib, SciPy), easy to use, simple syntax, open source nature - ported to many platforms, support RDBS, write Python code to access DBs via Python DB API
- Python  - Open Source, not proprietary
- Notebook - runs an environment that allows creating & sharing docs that contain live code, equations, visualizations, explanatory texts
- Notebook interface - virtual notebook environemnt used for programming (e.g: mathematical workbook, maple worksheet, matlab notebook, IPython Jupyter, R markdown, Apache Zeppelin, Apache Spark notebook, Databricks cloud)
- Jupyter notebook - open source web application let users share documents tht contain live code, equations, visualizations, narrative texts
- Pros of Jupyter Notebook:
    * support >40 prorgamming lang - Python, R, Julia, Scala
    * Share notebooks by email, dropbox, GitHub
    * interactive output (rich interactive output HTML, images, videos, LaTex)
    * big data integration (leverage Apache Spark from Python, R, Scala, other big data tools - scikit-learn, ggplot2, TensorFlow, pandas)
* API - set of functions that you can call to get access to some type of servers

**Writing code using DB-API**
* 2 concepts in the Python DB-API:
    * Connection objects - connect to DB and manage transactions
    * Query objects - cursor objects used for run queries
* Database cursor - control structure enables traversal over the records in a database like a file handler

**Connecting to a database using ibm_db API**
* ibm_db API library
    * provides SQL APIs for Python applciations includes functions for connecting to a DB, preparing and issuing SQL statements, fetching rows from result sets
    * includes functions for calling stored procedures, committing and rolling back transactions, handling errors and retrieving metadata
    * provides a variety of useful Python functions for accessing and manipulating data in an IBM data server database
    * can be used to connect to certain IBM databases like Db2 and Db2 Warehouse on Cloud

**Creating tables, loading data and querying data**
* `ibm_db.exec_immediate(conn, statement, optional)` - prepares and executes a SQL statement
* Pandas - retrieve data from tables, high level data structures, manipulate + data analysis

**Join Overview**
* JOIn operator - combine rows from 2 or more tables, based on relationship
* Inner Join - returns the rows that match only
* Outer Join
    * Left Outer Join = left join - displays all rows from the `left` table and combines matching rows from the `right` table
    * Right Outer Join = right join - all rows from the `right` table & any matching rows from the `left` table
    * Full Outer Join = full join - returns all rows from `both`tables rather than the rows that match
* Use alias instead of rewriting the whole table name

**Working with CSV file**
* spaces are mapped to underscores
* double underscrores
* trailing underscroe- end of query
* user backslash `\` as escape character in cases where query contains single quotes
* use `\` to split queries to multiple lines in Jupyter or use `%%sql` -> implies the rest of the content in the cell is to be interpreted by SQL magic