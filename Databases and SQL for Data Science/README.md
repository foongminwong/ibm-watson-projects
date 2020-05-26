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


