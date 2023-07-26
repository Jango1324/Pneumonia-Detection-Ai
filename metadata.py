from notebook import pkg

# get a table with information about ALL of our images
metadata = pkg.get_metadata()

# what does it look like?
print(metadata)
