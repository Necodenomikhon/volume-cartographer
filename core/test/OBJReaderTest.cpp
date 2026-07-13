#include <iostream>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "vc/core/io/OBJReader.hpp"
#include "vc/core/types/Exceptions.hpp"

using namespace volcart;

class OBJReader : public ::testing::Test
{
public:
    OBJReader() = default;
    io::OBJReader reader;
    std::string path{"vc_core_OBJReader_"};
};

TEST_F(OBJReader, TexturedWithNormals)
{
    reader.setPath(path + "TexturedWithNormals.obj");
    ASSERT_NO_THROW(reader.read());

    auto mesh = reader.getMesh();
    auto uv = reader.getUVMap();
    auto img = reader.getTextureMat();
    EXPECT_EQ(mesh->GetNumberOfPoints(), 16);
    EXPECT_EQ(mesh->GetNumberOfCells(), 18);
    EXPECT_FALSE(uv->empty());
    EXPECT_FALSE(img.empty());
}

TEST_F(OBJReader, Textured)
{
    reader.setPath(path + "Textured.obj");
    ASSERT_NO_THROW(reader.read());

    auto mesh = reader.getMesh();
    auto uv = reader.getUVMap();
    auto img = reader.getTextureMat();
    EXPECT_EQ(mesh->GetNumberOfPoints(), 16);
    EXPECT_EQ(mesh->GetNumberOfCells(), 18);
    EXPECT_FALSE(uv->empty());
    EXPECT_FALSE(img.empty());
}

TEST_F(OBJReader, UntexturedWithNormals)
{
    reader.setPath(path + "UntexturedWithNormals.obj");
    ASSERT_NO_THROW(reader.read());

    auto mesh = reader.getMesh();
    auto uv = reader.getUVMap();
    EXPECT_EQ(mesh->GetNumberOfPoints(), 16);
    EXPECT_EQ(mesh->GetNumberOfCells(), 18);
    EXPECT_TRUE(uv->empty());
}

TEST_F(OBJReader, VariableVertexInfo)
{
    reader.setPath(path + "VariableVertInfo.obj");
    ASSERT_NO_THROW(reader.read());

    auto mesh = reader.getMesh();
    auto uv = reader.getUVMap();
    auto img = reader.getTextureMat();
    EXPECT_EQ(mesh->GetNumberOfPoints(), 16);
    EXPECT_EQ(mesh->GetNumberOfCells(), 18);
    EXPECT_FALSE(uv->empty());
    EXPECT_FALSE(img.empty());
}

TEST_F(OBJReader, PointCloud)
{
    reader.setPath(path + "PointCloud.obj");
    ASSERT_NO_THROW(reader.read());

    auto mesh = reader.getMesh();
    EXPECT_EQ(mesh->GetNumberOfPoints(), 16);
    EXPECT_EQ(mesh->GetNumberOfCells(), 0);
}

TEST_F(OBJReader, Invalid)
{
    reader.setPath(path + "Invalid.obj");
    EXPECT_THROW(reader.read(), IOException);
}

// A vertex referenced with two different UV indices across its incident
// faces (i.e. a UV seam) must NOT be split into separate mesh points -
// mesh/cell connectivity has to stay based on position alone, since
// splitting disconnects faces that share an edge in the original topology
// and breaks algorithms (e.g. ABF++ flattening) that require the mesh to
// remain a connected manifold. The point-based UV map is expected to
// collapse to one (arbitrary, last-write-wins) value per point; exact
// per-corner UVs are recovered separately via getFaceUVs().
TEST_F(OBJReader, UVSeam)
{
    reader.setPath(path + "UVSeam.obj");
    ASSERT_NO_THROW(reader.read());

    auto mesh = reader.getMesh();
    auto uv = reader.getUVMap();
    auto faceUVs = reader.getFaceUVs();

    // No points are added for the seam - connectivity is unaffected
    EXPECT_EQ(mesh->GetNumberOfPoints(), 4);
    EXPECT_EQ(mesh->GetNumberOfCells(), 2);
    ASSERT_FALSE(uv->empty());

    // Both faces still reference the same point (1) for the seam vertex
    ITKCell::CellAutoPointer cell;
    mesh->GetCell(0, cell);
    EXPECT_EQ(cell->GetPointIdsContainer().GetElement(0), 0);
    EXPECT_EQ(cell->GetPointIdsContainer().GetElement(1), 1);
    EXPECT_EQ(cell->GetPointIdsContainer().GetElement(2), 2);

    mesh->GetCell(1, cell);
    EXPECT_EQ(cell->GetPointIdsContainer().GetElement(0), 1);
    EXPECT_EQ(cell->GetPointIdsContainer().GetElement(1), 3);
    EXPECT_EQ(cell->GetPointIdsContainer().GetElement(2), 2);

    // getFaceUVs() must still recover the correct, distinct UV for each
    // face's use of the shared vertex, indexed by cell
    ASSERT_EQ(faceUVs.size(), 2);
    EXPECT_EQ(faceUVs[0][0], cv::Vec2d(0.0, 0.0));  // face 0, corner 0 (v1)
    EXPECT_EQ(faceUVs[0][1], cv::Vec2d(1.0, 0.0));  // face 0, corner 1 (v2)
    EXPECT_EQ(faceUVs[0][2], cv::Vec2d(0.0, 1.0));  // face 0, corner 2 (v3)
    EXPECT_EQ(faceUVs[1][0], cv::Vec2d(5.0, 5.0));  // face 1, corner 0 (v2!)
    EXPECT_EQ(faceUVs[1][1], cv::Vec2d(1.0, 1.0));  // face 1, corner 1 (v4)
    EXPECT_EQ(faceUVs[1][2], cv::Vec2d(0.0, 1.0));  // face 1, corner 2 (v3)
}