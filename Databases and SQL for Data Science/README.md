# Databases and SQL for Data Science

## Introduction to SQL for Data Science
* SQL (Structured English Query Language) - a language used for relational database to query/get data out of a database
* Data - a collection of facts in the form of words, numbers or pictures, critical assets for business and collected practically (e.g.: address, phone #, account #, etc.)
* Database - a repository of data/a program that stores data which also provides the functionality for adding, modifying and querying data
* Relational database - data stored in tabular form (organzied in tables likes spreadsheet), can form relationaships between tables
* Database management system/DBMS - a set of software tools for data in database
* Relational database management system/RDBMS - a set of tools that controls the data such as access, organziation and storage (e.g: mySQL, Oracle Database, Db2 Warehouse on Cloud, DB2 Express-C)
* Basic SQL commands: CREATE, INSERT, SELECT, UPDATE, DELETE

## Create a Database Instance on Cloud (a service isntance on an IBM Db2 on cloud database)
* cloud database - database service built and accessed through a cloud platform (users install software on a cloud infrastructure to implement the database)
* Pros: 
    * Can access cloud databases from virtually anywhere using a vendors API or web interface 
    * Scalability - Can expand their storage capacities during runtimeto accommodate changing needs
    * Disaster recovery - kept data secure through backups on remote servers
    * e.g.: IBM DB2 on Cloud, COmpose for PostgreSQL, Oracle Database Cloud, Microsoft Azure Cloud, SQL Databse, Amazon Relational Database Services 

## CREATE TABLE Statement
* DDL (Data Definition Language) statements - define, change or drop data (e.g. CREATE)
* DML (Data Manipulation Language) statements: read & modify data
* Columns -> Entity attributes (e.g. char - character string of a fixed length, varcahr - character string of a variable length)
* Priamry key - no duplicate values can exist, uniquely identifies each tuple or row in a table, constraint prevents duplicate values in table
* Constraint NOT NULL - ensures fields cannot contain a null value