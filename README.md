Use Latin Hypercube sampling to generate simple (backwards) hadronic calorimeter geometry used to train a surrogate.

Also, the following converts a xml file into a viewable root file:
geoConverter -compact2tgeo -input geometries/generated/81c3da7d/geometry.xml -output display.root

edm4hep2json visuals/rune3896ec0d8.edm4hep.root -o visuals/rune3896ec0d8.edm4hep.json