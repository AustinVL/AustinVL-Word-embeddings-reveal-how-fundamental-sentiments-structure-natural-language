

The Stata analyses require Stata 16 or higher (uses Stata's frames).

The folder **do** contains .do files for analyses and the folder **raw** contains
the raw data used.

Running the .do file **MasterProjectWorkflow.do** will execute the .do files
needed to reproduce analyses from the raw data from the different .do files.
This .do file assumes the working directory has been set to the **do**
folder and uses relative paths so that it should not work with the
directory structure in this replication package regardless of whether it
has been placed.

Running this file will populate the empty folder in this package, rewriting
files contained therein as needed.  Specifically, the folder **derived** contains
datasets made in the course of cleaning and analyzing data from the -raw- data.
The **fig** and **table** folder create figures and tables in the paper.
