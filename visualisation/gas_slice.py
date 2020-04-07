

fig = plt.figure(1, figsize=(7,7))

med2 = np.median(coords[:,2])

s1 = np.fabs(coords[:,2]-med2)<1.0 # 1 Mpc wide slice

ncoords = coords[s1]

print(ncoords.shape)

qv = QuickView(ncoords, r='infinity', plot=False)

plt.imshow(qv.get_image(), extent=qv.get_extent(), origin='lower', cmap=cmaps.twilight())

fig.savefig('gas_slice.png')
