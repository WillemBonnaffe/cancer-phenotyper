#################################
## analyse-gland-annotations.r ##
#################################

## Goal: Analyse annotations of glands made by pathologists.
## Author: Willem Bonnaffe (w.bonnaffe@gmail.com)

##############
## INITIATE ##
##############

## Load data
dat = read.csv("GSG - Gland scores - 27.04.2026.csv", sep=",")
head(dat)
colnames(dat)

#
###

##############
## ANALYSIS ##
##############

# table(dat[,c("cluster", "approximate_grade_numeric")])
# plot(table(dat[,c("cluster", "approximate_grade_numeric")]), col=c("green","orange","red"))
# plot(table(dat[,c("cluster", "approximate_gleason_numeric")]), col=c("green","yellow","orange","red"))
# plot(table(dat[,c("cluster", "inflammatory_cells_numeric")]), col=c("green","yellow","orange"))

#
###

#####
## ##
#####

s1 = which(dat$annotator=="Richard Colling")
s2 = which(dat$annotator=="Clare Verrill")
r_grade = 1-mean((dat$approximate_grade_numeric[s1] - dat$approximate_grade_numeric[s2]) != 0)
r_gleason = 1-mean((dat$approximate_gleason_numeric[s1] - dat$approximate_gleason_numeric[s2])!=0)
r_inflammatory = 1-mean((dat$inflammatory_cells_numeric[s1] - dat$inflammatory_cells_numeric[s2])!=0)
print(r_grade)
print(r_gleason)
print(r_inflammatory)

#
###

######
##  ##
######

## Set general graphical parameters for consistency
par(mfrow=c(3,1),        # 3 plots side-by-side
    mar=c(5,5,4,2),      # margins
    cex.lab=1.4,
    cex.axis=1.2,
    cex.main=1.5,
    bty="l")
border_colour = "white"

tab <- t(table(dat$cluster, dat$approximate_grade_numeric))
tab <- tab[,c("C15", "C2", "C13", "C9")]
tab <- tab / 60 

barplot(tab,
        beside=TRUE,
        col=c("green","orange","red"),
        border=border_colour,
        xlab="Cluster",
        ylab="Proportion of Glands",
        main="Cluster vs Risk Category")

legend("topright",
       legend=c("Low","Intermediate","High"),
       fill=c("green","orange","red"),
       title="Risk Category",
       cex=1,
       bty="n")

tab <- t(table(dat$cluster, dat$approximate_gleason_numeric))
tab <- tab[,c("C15", "C2", "C13", "C9")]
tab <- tab / 60 

barplot(tab,
        beside=TRUE,
        col=c("green","yellow","orange","red"),
        border=border_colour,
        xlab="Cluster",
        ylab="Proportion of Glands",
        main="Cluster vs Gleason")

legend("topright",
       legend=c("3+3","3+4","4+3","4+4"),
       fill=c("green","yellow","orange","red"),
       title="Approximate Gleason",
       cex=1,
       bty="n")

tab <- t(table(dat$cluster, dat$inflammatory_cells_numeric))
tab <- tab[,c("C15", "C2", "C13", "C9")]
tab <- tab / 60 


barplot(tab,
        beside=TRUE,
        col=c("green","yellow"),
        border=border_colour,
        xlab="Cluster",
        ylab="Proportion of Glands",
        main="Cluster vs Inflammatory Cells")

legend("topright",
       legend=c("Absent","Present"),
       fill=c("green","yellow"),
       title="Inflammatory Cells",
       cex=1,
       bty="n")

#
###