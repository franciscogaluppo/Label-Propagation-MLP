library(ggplot2)

data <- read.csv("report.csv")
data$p <- as.factor(data$p)

pdf("density.pdf", width=8, height=3)
ggplot(data, aes(x=frac, color=p)) +
	geom_density(size=1) +
	scale_color_brewer(palette="Dark2") +
	theme_classic() +
	xlab("Fraction of the largest label") +
	ylab("Density")
dev.off()

pdf("accuracy.pdf", width=8, height=3)
ggplot(data, aes(y=test_acc, x=p, fill=p)) +
	geom_boxplot() +
	geom_hline(yintercept=0.1, linetype="dashed", color = "black") +
	scale_fill_brewer(palette="Dark2") +
	theme_classic() +
	xlab("p") +
	ylab("Test accuracy")
dev.off()

means <- as.data.frame.table(with(data, tapply(test_acc, p, mean))) 
names(means) <- c("p", "test_acc")
means$p <- as.factor(means$p)
pdf("bars.pdf", width=8, height=3)
ggplot(means, aes(x=p, y=test_acc, fill=p)) +
	geom_bar(stat = "identity") +
	geom_hline(yintercept=0.1, linetype="dashed", color = "black") +
	scale_fill_brewer(palette="Dark2") +
	theme_classic() +
	xlab("p") +
	ylab("Mean test accuracy")
dev.off()

pdf("scatter.pdf", width=8, height=3)
ggplot(data, aes(x=frac, y=test_acc, color=p)) +
	geom_point() +
	geom_abline(intercept=0, slope=1, linetype="dashed", color = "black") +
	scale_color_brewer(palette="Dark2") +
	theme_classic() +
	xlab("Fraction of the largest label") +
	ylab("Test accuracy")
dev.off()
