import cvcorelib
import pycc

CC = pycc.GetInstance()


def doSubSampling(pc):
    refcloud = cvcorelib.CloudSamplingTools.subsampleCloudRandomly(pc, pc.size() // 2)
    randomPc = pc.partialClone(refcloud)
    randomPc.setName("Randomly subsampled")
    CC.addToDB(randomPc)

    refcloud = cvcorelib.CloudSamplingTools.subsampleCloudWithOctree(pc, pc.size() // 4,
                                                                     cvcorelib.CloudSamplingTools.RANDOM_POINT)
    randomPc = pc.partialClone(refcloud)
    randomPc.setName("Subsampled using octree (RANDOM_POINT)")
    CC.addToDB(randomPc)

    refcloud = cvcorelib.CloudSamplingTools.subsampleCloudWithOctree(pc, pc.size() // 4,
                                                                     cvcorelib.CloudSamplingTools.NEAREST_POINT_TO_CELL_CENTER)
    randomPc = pc.partialClone(refcloud)
    randomPc.setName("Subsampled using octree (NEAREST_POINT_TO_CELL_CENTER)")
    CC.addToDB(randomPc)


def main():
    entities = CC.getSelectedEntities()
    print(f"Selected entities: {entities}")

    if not entities:
        raise RuntimeError("No entities selected")

    pointCloud = entities[0]
    print(pointCloud)

    pycc.RunInThread(doSubSampling, pointCloud)

    CC.updateUI()


if __name__ == '__main__':
    main()
