set experimental_fulltext_index=1;
use mo_catalog;
create cluster table clu01(col1 int, col2 decimal);
insert into clu01 values(1,2,0);

drop snapshot if exists sp01;
create snapshot sp01 for account;
insert into clu01 values(2,3,0);

restore account sys from snapshot sp01;

select * from clu01;
drop table clu01;




drop database if exists test;
create database test;
use test;
create table clu01(col1 int, col2 decimal);
insert into clu01 values(1,2);

drop snapshot if exists sp01;
create snapshot sp01 for account;
insert into clu01 values(2,3);

restore account sys from snapshot sp01;

select * from clu01;
select count(*) from clu01;
drop table clu01;
drop database test;




-- sys account restore to account: single db, single table
drop database if exists test01;
create database test01;
use test01;

drop table if exists rs01;
create table rs01 (col1 int, col2 decimal(6), col3 varchar(30));
insert into rs01 values (1, null, 'database');
insert into rs01 values (2, 38291.32132, 'database');
insert into rs01 values (3, null, 'database management system');
insert into rs01 values (4, 10, null);
insert into rs01 values (1, -321.321, null);
insert into rs01 values (2, -1, null);
select count(*) from rs01;

drop snapshot if exists sp01;
create snapshot sp01 for account;
select count(*) from rs01 {snapshot = 'sp01'};
insert into rs01 values (2, -1, null);
insert into rs01 values (1, -321.321, null);
select * from rs01;

select count(*) from mo_catalog.mo_tables{snapshot = 'sp01'} where reldatabase = 'test01';
-- @ignore:0,6,7
select * from mo_catalog.mo_database{snapshot = 'sp01'} where datname = 'test01';
select attname from mo_catalog.mo_columns{snapshot = 'sp01'} where att_database = 'test01';
restore account sys from snapshot sp01;
select count(*) from rs01 {snapshot = 'sp01'};
select * from rs01 {snapshot = 'sp01'};
select count(*) from mo_catalog.mo_tables{snapshot = 'sp01'} where reldatabase = 'test01';
-- @ignore:0,6,7
select * from mo_catalog.mo_database{snapshot = 'sp01'} where datname = 'test01';
select attname from mo_catalog.mo_columns{snapshot = 'sp01'} where att_database = 'test01';
drop snapshot sp01;
drop database test01;




-- sys account restore to account: single db, multi table
use mo_catalog;
drop table if exists cluster01;
create cluster table cluster01(col1 int,col2 bigint);
insert into cluster01 values(1,2,0);
insert into cluster01 values(2,3,0);
select * from cluster01;
drop database if exists test02;
create database test02;
use test02;
drop table if exists rs02;
create table rs02 (col1 int, col2 datetime);
insert into rs02 values (1, '2020-10-13 10:10:10');
insert into rs02 values (2, null);
insert into rs02 values (1, '2021-10-10 00:00:00');
insert into rs02 values (2, '2023-01-01 12:12:12');
insert into rs02 values (2, null);
insert into rs02 values (3, null);
insert into rs02 values (4, '2023-11-27 01:02:03');
select * from rs02;
drop table if exists rs03;
create table rs03 (col1 int, col2 float, col3 decimal, col4 enum('1','2','3','4'));
insert into rs03 values (1, 12.21, 32324.32131, 1);
insert into rs03 values (2, null, null, 2);
insert into rs03 values (2, -12.1, 34738, null);
insert into rs03 values (1, 90.2314, null, 4);
insert into rs03 values (1, 43425.4325, -7483.432, 2);
drop snapshot if exists sp02;
create snapshot sp02 for account;
select count(*) from mo_catalog.mo_tables{snapshot = 'sp02'} where reldatabase = 'test02';
-- @ignore:0,5,6,7
select * from mo_catalog.mo_database{snapshot = 'sp02'} where datname = 'test02';
-- @ignore:0,5,6,7
select * from mo_catalog.mo_database{snapshot = 'sp02'} where datname = 'mo_catalog';

select attname from mo_catalog.mo_columns{snapshot = 'sp02'} where att_database = 'test02';

use mo_catalog;
insert into cluster01 values(100,2,0);
insert into cluster01 values(200,3,0);
select count(*) from cluster01;
select count(*) from cluster01{snapshot = 'sp02'};

use test02;
insert into rs02 select * from rs02;
select count(*) from rs02;
select count(*) from rs02{snapshot = 'sp02'};

delete from rs03 where col1 = 1;
select count(*) from rs03;
select count(*) from rs03{snapshot = 'sp02'};

restore account sys from snapshot sp02;

show databases;
select count(*) from rs02;
select count(*) from rs03;
use mo_catalog;
select count(*) from cluster01;
drop table cluster01;
use test02;
drop table rs02;
drop table rs03;
drop snapshot sp02;
drop database test02;




-- table with foreign key restore
drop database if exists test03;
create database test03;
use test03;
drop table if exists pri01;
create table pri01(
                     deptno int unsigned comment '部门编号',
                     dname varchar(15) comment '部门名称',
                     loc varchar(50)  comment '部门所在位置',
                     primary key(deptno)
) comment='部门表';

insert into pri01 values (10,'ACCOUNTING','NEW YORK');
insert into pri01 values (20,'RESEARCH','DALLAS');
insert into pri01 values (30,'SALES','CHICAGO');
insert into pri01 values (40,'OPERATIONS','BOSTON');

drop table if exists aff01;
create table aff01(
                    empno int unsigned auto_increment COMMENT '雇员编号',
                    ename varchar(15) comment '雇员姓名',
                    job varchar(10) comment '雇员职位',
                    mgr int unsigned comment '雇员对应的领导的编号',
                    hiredate date comment '雇员的雇佣日期',
                    sal decimal(7,2) comment '雇员的基本工资',
                    comm decimal(7,2) comment '奖金',
                    deptno int unsigned comment '所在部门',
                    primary key(empno),
                    constraint `c1` foreign key (deptno) references pri01 (deptno)
);

insert into aff01 values (7369,'SMITH','CLERK',7902,'1980-12-17',800,NULL,20);
insert into aff01 values (7499,'ALLEN','SALESMAN',7698,'1981-02-20',1600,300,30);
insert into aff01 values (7521,'WARD','SALESMAN',7698,'1981-02-22',1250,500,30);
insert into aff01 values (7566,'JONES','MANAGER',7839,'1981-04-02',2975,NULL,20);
insert into aff01 values (7654,'MARTIN','SALESMAN',7698,'1981-09-28',1250,1400,30);
insert into aff01 values (7698,'BLAKE','MANAGER',7839,'1981-05-01',2850,NULL,30);
insert into aff01 values (7782,'CLARK','MANAGER',7839,'1981-06-09',2450,NULL,10);
insert into aff01 values (7788,'SCOTT','ANALYST',7566,'0087-07-13',3000,NULL,20);
insert into aff01 values (7839,'KING','PRESIDENT',NULL,'1981-11-17',5000,NULL,10);
insert into aff01 values (7844,'TURNER','SALESMAN',7698,'1981-09-08',1500,0,30);
insert into aff01 values (7876,'ADAMS','CLERK',7788,'0087-07-13',1100,NULL,20);
insert into aff01 values (7900,'JAMES','CLERK',7698,'1981-12-03',950,NULL,30);
insert into aff01 values (7902,'FORD','ANALYST',7566,'1981-12-03',3000,NULL,20);
insert into aff01 values (7934,'MILLER','CLERK',7782,'1982-01-23',1300,NULL,10);

select count(*) from pri01;
select count(*) from aff01;

show create table pri01;
show create table aff01;

select count(*) from mo_catalog.mo_tables{snapshot = 'sp04'} where reldatabase = 'test03';
-- @ignore:0,6,7
select * from mo_catalog.mo_database{snapshot = 'sp04'} where datname = 'test03';
select attname from mo_catalog.mo_columns{snapshot = 'sp04'} where att_database = 'test03';

drop snapshot if exists sp04;
create snapshot sp04 for account;
-- @ignore:1
show snapshots where snapshot_name = 'sp04';

select * from aff01{snapshot = 'sp04'};
select * from pri01{snapshot = 'sp04'};

drop database test03;
select * from test03.aff01{snapshot = 'sp04'};
select * from test03.pri01{snapshot = 'sp04'};
select count(*) from test03.aff01{snapshot = 'sp04'};

restore account sys from snapshot sp04;
use test03;
show create table aff01;
show create table pri01;
select count(*) from aff01;
drop database test03;
drop snapshot sp04;




-- restore account to current account
drop database if exists test01;
create database test01;
use test01;
create table t1(col1 int, col2 decimal);
insert into t1 values(1,2);
insert into t1 values(2,3);
insert into t1 values(3,4);
create table t2(cool1 int primary key , col2 decimal);
insert into t2 select * from t1;
create table t3 like t2;
select count(*) from t1;
select count(*) from t2;
select count(*) from t3;

drop database if exists test02;
create database test02;
use test02;
create table t1(col1 int, col2 decimal);
insert into t1 values(1,2);
insert into t1 values(2,3);
insert into t1 values(3,4);
create table t2(col1 int primary key , col2 decimal);
insert into t2 select * from t1;
create table t3 like t2;
insert into t3 select * from t2;
select count(*) from t1;
select count(*) from t2;
select count(*) from t3;

drop database if exists test03;
create database test03;
use test03;
create table t1(col1 int, col2 decimal);
insert into t1 values(1,2);
insert into t1 values(2,3);
insert into t1 values(3,4);
create table t2(cool1 int primary key , col2 decimal);
insert into t2 select * from t1;
create table t3 like t2;
insert into t3 select * from t2;
insert into t3 select * from t2;
select count(*) from t1;
select count(*) from t2;
select count(*) from t3;

drop snapshot if exists snap01;
create snapshot snap01 for account;
select count(*) from test01.t1 {snapshot = 'snap01'};
select count(*) from test02.t2 {snapshot = 'snap01'};
select count(*) from test03.t3 {snapshot = 'snap01'};

drop database test01;
drop database test02;
show databases;

select * from test01.t1;
select count(*) from test03.t3;

restore account sys from snapshot snap01;

show databases;
select count(*) from test01.t1;
select * from test01.t1;
select count(*) from test02.t2;
select * from test02.t2;
select count(*) from test03.t3;
select * from test03.t3;
show create table test01.t1;
show create table test02.t2;
show create table test03.t2;
drop database test01;
drop database test02;
drop database test03;
drop snapshot snap01;




-- restore null
drop snapshot if exists sp05;
create snapshot sp05 for account;
create database db01;
restore account sys FROM snapshot sp05;
show databases;
drop snapshot sp05;




-- sys create sp01,sp02, restore sp02, restore sp01
drop database if exists db01;
create database db01;
use db01;
drop table if exists table01;
create table table01(col1 int primary key , col2 decimal, col3 char, col4 varchar(20), col5 text, col6 double);
insert into table01 values (1, 2, 'a', '23eiojf', 'r23v324r23rer', 3923.324);
insert into table01 values (2, 3, 'b', '32r32r', 'database', 1111111);
drop table if exists table02;
create table table02 (col1 int unique key, col2 varchar(20));
insert into table02 (col1, col2) values (133, 'database');

drop snapshot if exists sp07;
create snapshot sp07 for account;

drop table table01;
insert into table02 values(134, 'database');

drop snapshot if exists sp08;
create snapshot sp08 for account;
-- @ignore:1
show snapshots;
restore account sys from snapshot sp08;
select * from table02;
select * from db01.table01;
select count(*) from table02;

restore account sys from snapshot sp07;

select * from table01;
select * from table02;
select count(*) from table01;
select count(*) from table02;

drop snapshot sp07;
drop snapshot sp08;
drop database db01;




-- sys create sp01,sp02, restore sp01, restore sp02
drop database if exists db02;
create database db02;
use db02;
drop table if exists table01;
create table table01(col1 int primary key , col2 decimal, col3 char, col4 varchar(20), col5 text, col6 double);
insert into table01 values (1, 2, 'a', '23eiojf', 'r23v324r23rer', 3923.324);
insert into table01 values (2, 3, 'b', '32r32r', 'database', 1111111);
drop table if exists table02;
create table table02 (col1 int unique key, col2 varchar(20));
insert into table02 (col1, col2) values (133, 'database');

drop snapshot if exists sp09;
create snapshot sp09 for account;

alter table table01 drop primary key;
insert into table02 values(134, 'database');

drop snapshot if exists sp10;
create snapshot sp10 for account;
-- @ignore:1
show snapshots;
restore account sys from snapshot sp09;
select * from table02;
select * from db02.table01;
select count(*) from table02;
select count(*) from table01;

restore account sys from snapshot sp10;
select * from db02.table01;
select count(*) from table01;
show create table db02.table01;
select * from db02.table02;
select count(*) from table02;

drop snapshot sp09;
drop snapshot sp10;
drop database db02;




-- restore frequently
drop database if exists db03;
create database db03;
use db03;
drop table if exists ti1;
drop table if exists tm1;
drop table if exists ti2;
drop table if exists tm2;

create  table ti1(a INT not null, b INT, c INT);
create  table tm1(a INT not null, b INT, c INT);
create  table ti2(a INT primary key AUTO_INCREMENT, b INT, c INT);
create  table tm2(a INT primary key AUTO_INCREMENT, b INT, c INT);
show create table ti1;
show create table tm1;
show create table ti2;
show create table tm2;
drop snapshot if exists sp11;
create snapshot sp11 for account;

insert into ti1 values (1,1,1), (2,2,2);
insert into ti2 values (1,1,1), (2,2,2);
select * from ti1;
select * from tm1;
select * from ti2;
select * from tm2;
drop snapshot if exists sp12;
create snapshot sp12 for account;

insert into tm1 values (1,1,1), (2,2,2);
insert into tm2 values (1,1,1), (2,2,2);
select * from ti1 {snapshot = 'sp12'};
select * from tm1;
select * from ti2 {snapshot = 'sp12'};
select * from tm2;
drop snapshot if exists sp13;
create snapshot sp13 for account;

alter table ti1 add constraint fi1 foreign key (b) references ti2(a);
alter table tm1 add constraint fm1 foreign key (b) references tm2(a);
drop snapshot if exists sp14;
create snapshot sp14 for account;

show create table ti1 {snapshot = 'sp14'};
show create table tm1 {snapshot = 'sp13'};
show create table ti1 {snapshot = 'sp14'};
show create table tm1 {snapshot = 'sp13'};

alter table ti1 drop foreign key fi1;
alter table tm1 drop foreign key fm1;
truncate ti2;
truncate tm2;
drop snapshot if exists sp15;
create snapshot sp15 for account;

show create table ti1 {snapshot = 'sp14'};
show create table tm1 {snapshot = 'sp15'};
show create table ti1 {snapshot = 'sp14'};
show create table tm1 {snapshot = 'sp15'};

select count(*) from ti1;
select count(*) from tm1;
select count(*) from ti2;
select count(*) from tm2;

restore account sys from snapshot sp11;
show databases;
select * from db03.ti1;
select * from db03.tm1;
select * from db03.ti2;
select * from db03.tm2;
show create table db03.ti1;
show create table db03.tm1;
show create table db03.ti2;
show create table db03.tm2;

restore account sys from snapshot sp14;
show databases;
select * from db03.ti1;
select * from db03.tm1;
select * from db03.ti2;
select * from db03.tm2;
show create table db03.ti1;
show create table db03.tm1;
show create table db03.ti2;
show create table db03.tm2;

drop database db03;
drop snapshot sp15;
drop snapshot sp14;
drop snapshot sp13;
drop snapshot sp12;
drop snapshot sp11;




-- restore db to current account
drop database if exists db04;
create database db04;
use db04;
create table table01(col1 int primary key );
insert into table01 values (1);
insert into table01 values (2);
drop database if exists db05;
create database db05;
use db05;
create table table01(col1 int primary key );
insert into table01 values (1);
insert into table01 values (2);
drop database if exists db06;
create database db06;
use db06;
create table table01(col1 int primary key );
insert into table01 values (1);
insert into table01 values (2);
select * from db04.table01;
select * from db05.table01;
select * from db06.table01;

drop snapshot if exists sp13;
create snapshot sp13 for account;
insert into db04.table01 values (200);
insert into db05.table01 values (400);

restore account sys database db04 from snapshot sp13;

show databases;
use db04;
select count(*) from db04.table01;
select * from db04.table01;
use db05;
select count(*) from db05.table01;
select * from db05.table01;
use db06;
select count(*) from db06.table01;
select * from db06.table01;
drop snapshot sp13;
drop database db04;
drop database db05;
drop database db06;




-- restore db to current account
drop database if exists db07;
create database db07;
use db07;
create table table01 (col1 int, col2 enum ('a','b','c'));
insert into table01 values(1,'a');
insert into table01 values(2, 'b');
create table table02 (col1 int, col2 enum ('a','b','c'));
insert into table02 values(1,'a');
insert into table02 values(2, 'b');
drop database if exists db08;
create database db08;
use db08;
create table index01(col1 int,key key1(col1));
insert into index01 values (1);
insert into index01 values (2);
show create table db07.table02;

drop snapshot if exists sp14;
create snapshot sp14 for account;

use db07;
drop table table01;
alter table table02 add column newC int first;
show create table table02;
show create table table02 {snapshot = 'sp14'};
drop database db08;

drop snapshot if exists sp15;
create snapshot sp15 for account;

restore account sys database db07 from snapshot sp15;
restore account sys database db08 from snapshot sp15;

show databases;
use db07;
show tables;
show create table table01 {snapshot = 'sp14'};
show create table table01;
select * from table01;

-- @ignore:1
show snapshots;
drop snapshot sp14;
drop snapshot sp15;
drop database db07;
drop database db08;




-- restore non-exists db to current account
drop database if exists db08;
create database db08;
use db08;
create table db08 (col1 int);
insert into db08 values(1), (100), (20000);
drop snapshot if exists sp15;
create snapshot sp15 for account;
insert into db08 (col1) values (3000);

restore account sys database db from snapshot sp15;
drop snapshot sp15;
drop database db08;




-- restore non-exists table to current account
drop database if exists db09;
create database db09;
use db09;
drop table if exists index01;
create table index01(
        col1 int not null,
        col2 date not null,
        col3 varchar(16) not null,
        col4 int unsigned not null,
        primary key (col1)
);
insert into index01 values(1, '1980-12-17','Abby', 21);
insert into index01 values(2, '1981-02-20','Bob', 22);
insert into index01 values(3, '1981-02-20','Bob', 22);
select count(*) from index01;

drop table if exists index02;
create table index02(col1 char, col2 int, col3 binary);
insert into index02 values('a', 33, 1);
insert into index02 values('c', 231, 0);
alter table index02 add key pk(col1) comment 'primary key';
select count(*) from index02;

drop database if exists db10;
create database db10;
use db10;
-- @bvt:issue#16438
drop table if exists index03;
create table index03 (
        emp_no      int             not null,
        birth_date  date            not null,
        first_name  varchar(14)     not null,
        last_name   varchar(16)     not null,
        gender      varchar(5)      not null,
        hire_date   date            not null,
        primary key (emp_no)
) partition by range columns (emp_no)(
    partition p01 values less than (100001),
    partition p02 values less than (200001),
    partition p03 values less than (300001),
    partition p04 values less than (400001)
);

insert into index03 values (9001,'1980-12-17', 'SMITH', 'CLERK', 'F', '2008-12-17'),
                          (9002,'1981-02-20', 'ALLEN', 'SALESMAN', 'F', '2008-02-20');
-- @bvt:issue
drop snapshot if exists sp16;
create snapshot sp16 for account;

use db09;
delete from index02 where col1 = 'a';
select * from index02;
update index01 set col1 = 1000 where col1 = 1;
select * from index01;

use db10;
-- @bvt:issue#16438
truncate index03;
-- @bvt:issue

restore account sys database db09 table index01 from snapshot sp16;
-- @bvt:issue#16438
restore account sys database db10 table index03 from snapshot sp16;
-- @bvt:issue

use db09;
select * from index02;
select * from index01;
show create table index01;
show create table index02;
use db10;
-- @bvt:issue#16438
select * from index03;
-- @bvt:issue

-- @ignore:1
show snapshots;
drop database db09;
drop database db10;
drop snapshot sp16;




-- cross restore non-exists table to current account
drop database if exists db11;
create database db11;
use db11;
drop table if exists pri01;
create table pri01 (col1 int, col2 decimal);
insert into pri01 (col1, col2) values (1,2378.328839842);
insert into pri01 values (234, -3923.2342342);
select * from pri01;

drop snapshot if exists sp17;
create snapshot sp17 for account;

alter table pri01 add constraint primary key(col1);
insert into pri01 values (23423, 32432543.3242);
insert into pri01 values (234222, -3923.2342342);

drop snapshot if exists sp18;
create snapshot sp18 for account;

restore account sys database db11 table pri01 from snapshot sp18;
show create table pri01;
select * from pri01;
select count(*) from pri01;

restore account sys database db11 table pri01 from snapshot sp17;
show create table pri01;
select * from pri01;
select count(*) from pri01;

restore account sys database db11 table pri01 from snapshot sp18;
show create table pri01;
select * from pri01;
select count(*) from pri01;
insert into pri01 values(23420, 32432543.3242);
select * from pri01;
show columns from pri01;
select count(*) from mo_catalog.mo_tables{snapshot = 'sp18'} where reldatabase = 'db11';
-- @ignore:0,6,7
select * from mo_catalog.mo_database{snapshot = 'sp18'} where datname = 'db11';
select attname from mo_catalog.mo_columns{snapshot = 'sp18'} where att_database = 'db11';

drop database db11;
drop snapshot sp18;
drop snapshot sp17;




drop database if exists test_fulltext_restore;
create database test_fulltext_restore;
use test_fulltext_restore;
drop table if exists posts;
drop table if exists comments;
create table posts (
    post_id int auto_increment primary key,
    title varchar(255),
    content text
);
create table comments (
    comment_id int auto_increment primary key,
    post_id int,
    comment_text text,
    foreign key (post_id) references posts(post_id)
);
drop snapshot if exists sp_fulltext;
create snapshot sp_fulltext for account;
alter table posts add fulltext(content);
insert into posts (title, content) values
('MO全文索引入门', 'MO全文索引是一种强大的工具，可以帮助你快速检索数据库中的文本数据。'),
('深入理解全文索引', '全文索引不仅可以提高搜索效率，还可以通过JOIN操作与其他表结合使用。');
insert into comments (post_id, comment_text) values
(1, '这篇文章很有用，谢谢分享！'),
(1, '我也在学习全文索引，很有帮助。'),
(2, '全文索引真的很强大，学习了。');
restore account sys from snapshot sp_fulltext;
show create table posts;
select * from posts;
select * from comments;
drop table comments;
drop table posts;
drop database test_fulltext_restore;
drop snapshot sp_fulltext;
-- @ignore:1
show snapshots;
