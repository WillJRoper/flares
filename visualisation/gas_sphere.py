

fig = plt.figure(1, figsize=(10,10),dpi=400,)

plt.axis('off')

med2 = np.median(coords[:,2])


coords_rec = coords - np.median(coords[:,:], axis=0)
r = np.linalg.norm(ncoords, axis=1)
s = r<14/0.7


qv = QuickView(coords_rec[s], r='infinity', plot=False)


img = qv.get_image()
print(np.min(img), np.max(img))

plt.imshow(img, extent=qv.get_extent(), origin='lower', cmap=cmaps.twilight())

fig.savefig('gas_sphere.png')
