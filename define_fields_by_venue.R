## make_field
## input: item-venue
## output: id-fieds, multiple ids 

library(stringr)
library(dplyr)
#library(jsonlite)
setwd("/Users/ZLab/Downloads")
#inputdata
item <- read.csv('./aminer/item_final.csv')

venue_words <- str_split(item$venue,pattern = '[:blank:]')
phys_materirals <- c('Physics','Optical','Nuclear','Electronics','Materials','Cailiao','Optics','Particle',
                     'Physica','Laser','Solid','Particles','Gravitation','Organic','Cosmology','Guangxue','Lasers',
                     'Particles','Atomic','Lixue','Optoelectronics','Nanotechnology','Spectroscopy',
                     'Chromatography','Radiology','Optica','Nano','Waves','Microwave','photonics','Superconductivity',
                     'Photonica')
chem <- c('Chemistry','Chemical','Molecular','Polymer','Biochemistry','Biochemical','Crystallographica','Crystal',
          'Polymers','Crystals','Tetrahedron','Thermophysics','Petrochemical','Polymeric','Gaofenzi','Macromolecular')
life_med <- c('Medical','Clinical','Medicine','Oncology','Biomedical','Genetics','Pharmaceutical','Biological',
              'Cell','Microbiology','Biology','Biotechnology','Immunology','Ecology','Gastroenterology',
              'Pharmacology','Neuroscience','Cellular','Ophthalmology','Medicinal','Surgery','Toxicology',
              'Rehabilitation','Drugs','Pathology','Diseases','Health','Rehabilitative','Therapy','Biophysical',
              'Zhongyao','Disease','Neurology','Digestology','Tumor','Pharmacological','Virology','Biomolecular',
              'Dermatology','Biophysics','Surgical','Brain','Care','Herbal','Nutrition')
info <- c('Computer','Information','Intelligence','Computing','Automation','Artificial','Electrical',
          'Xitong','Computational','Intelligent','Dianli','Jisuanji','Zidonghua','Simulation','Dianji','Software',
          'Kongzhi','Telecommunications','Computers','Dianzi','Circuits','Xinxi','Cybernetics','Semiconductor','Digital',
          'Tongxin','Web','Network','Semiconductors')
math <- c('Mathematics','Mathematical')
environ <- c('Environment','Enviromental','Earth','Geoscience','Ecologica','Statistical')

fields <- data.frame()

for(i in 1: length(venue_words)){
        vw <- venue_words[[i]]
        filedlist <- c('phys','chem','life','info','math','environ')
        pass <- c(sum(vw %in% phys_materirals)>0, sum(vw %in% chem)>0,sum(vw %in% life_med)>0,
                  sum(vw %in% info)>0, sum(vw %in% math)>0, sum(vw %in% environ)>0)
        field <- filedlist[pass]
        if(length(field)>0){
                df <- data.frame(field)
                df$id <- item$id[i]
                names(df) <- c('fileds','id')
                fields <- rbind(fields,df)
        }
        cat(i,'\n')
}

write.csv(fields,file='fields.csv')