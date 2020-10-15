library(ggplot2)
library(RColorBrewer)

# setting theme; want simple white background
theme_set(
  theme_minimal() +
    theme(legend.position = "right")
)

# scatterplot
q = ggplot(x, aes(Sepal.Length, Sepal.Width)) 

sq = q + geom_point(aes(color = Species), size=2.2) + theme(legend.position = "top") 

sq +  geom_smooth(aes(color = Species, fill = Species), method = "lm") + 
  scale_color_brewer(palette = "Dark2") + scale_fill_brewer(palette="Dark2")

  
# boxplot
p = ggplot(x, aes(x=Species, y=Sepal.Length))

bp =  p + geom_boxplot(aes(fill=Species)) + theme(legend.position = "top")

bp + scale_fill_brewer(palette = "Set2")


# density plot
c = ggplot(iris, aes(x=Sepal.Length))

dc = c + geom_density(kernel="gaussian", aes(color=Species, fill=Species), alpha=0.4) +
    geom_vline(data=mu_df, aes(xintercept=grmean, color=Species),linetype="dashed") +
    theme(legend.position = "top")

dc + scale_color_brewer(palette="Dark2") + scale_fill_brewer(palette = "Dark2")


#### Trouble recognizing input in ggplot(aes); Tidyverse edit
mu = ddply(iris, 'Species', summarise, grp.mean=mean(Sepal.Length))
library(tidyverse)
mu_df <- iris %>%
  group_by(Species) %>%
  summarise(grmean=mean(Sepal.Length))
mu_df
