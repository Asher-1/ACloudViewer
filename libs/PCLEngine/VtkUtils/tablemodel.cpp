#include "tablemodel.h"
#include "utils.h"

// CV_CORE_LIB
#include <CVLog.h>

// ECV_DB_LIB
#include <ecvHObject.h>
#include <ecvGenericMesh.h>
#include <ecvHObjectCaster.h>
#include <ecvGenericPointCloud.h>

// QT
#include <QDebug>

namespace VtkUtils
{

TableModel::TableModel(int column, int row, QObject* parent) 
	: QAbstractTableModel(parent), m_cols(column), m_rows(row)
{
	m_cols = column;
	m_rows = row;

	// m_data
	for (int i = 0; i < m_rows; i++) {
		QVector<qreal>* dataVec = new QVector<qreal>(m_cols);
		for (int k = 0; k < dataVec->size(); k++) {
			if (k % 2 == 0)
				dataVec->replace(k, i * 50 + qrand() % 20);
			else
				dataVec->replace(k, qrand() % 100);
		}
		m_data.append(dataVec);
	}
}

TableModel::TableModel(const ccHObject * objContainer, QObject * parent)
	: QAbstractTableModel(parent)
{
	updateData(objContainer);
}

void TableModel::updateData(const ccHObject * objContainer)
{
	clear();
	m_cols = 3;

	for (unsigned ci = 0; ci != objContainer->getChildrenNumber(); ++ci)
	{
		ccHObject* obj = objContainer->getChild(ci);
		ccGenericPointCloud* inputCloud = ccHObjectCaster::ToGenericPointCloud(obj);
		const ccGenericPointCloud::VisibilityTableType& verticesVisibility = inputCloud->getTheVisibilityArray();
		bool visFiltering = (verticesVisibility.size() >= inputCloud->size());
		int count = static_cast<int>(inputCloud->size());
		bool showColor = inputCloud->colorsShown();

		for (int i = 0; i < count; ++i)
		{
			if (visFiltering && verticesVisibility.at(i) != POINT_VISIBLE)
			{
				continue;
			}
			
			QVector<qreal>* dataVec = new QVector<qreal>(m_cols);
			const CCVector3* P = inputCloud->getPoint(static_cast<unsigned>(i));
			for (int k = 0; k < dataVec->size(); k++) {
				dataVec->replace(k, P->u[k]);
			}
			m_data.append(dataVec);
		}

		if (obj->isKindOf(CV_TYPES::POINT_CLOUD))
		{		
		}
		else if (obj->isKindOf(CV_TYPES::MESH))
		{
			ccGenericMesh* mesh = static_cast<ccGenericMesh*>(obj);
			//vertices visibility
			int triNum = static_cast<int>(mesh->size());
			for (int i = 0; i < triNum; ++i)
			{
				const cloudViewer::VerticesIndexes* tsi = mesh->getTriangleVertIndexes(i);
				if (visFiltering)
				{
					//we skip the triangle if at least one vertex is hidden
					if ((verticesVisibility[tsi->i1] != POINT_VISIBLE) ||
						(verticesVisibility[tsi->i2] != POINT_VISIBLE) ||
						(verticesVisibility[tsi->i3] != POINT_VISIBLE))
					{
						continue;
					}
				}
				m_vertices.append(Tuple3ui(tsi->i));
			}
		}
	}

	m_rows = m_data.size();
}

TableModel::~TableModel()
{
	clear();
}

void TableModel::random(int min, int max)
{
	m_randomMin = min;
	m_randomMax = max;

	for (int r = 0; r < m_rows; ++r) {
		for (int c = 0; c < m_cols; ++c) {
			auto vecPtr = m_data[r];
			vecPtr->replace(c, Utils::random(m_randomMin, m_randomMax));
		}
	}
	emit layoutChanged();
}

void TableModel::resize(int column, int row)
{
	if (m_cols == column && m_rows == row) {
//        qDebug() << "TableModel::resize: same cols & rows.";
		return;
	}

	if (m_cols != column)
		emit columnsChanged(m_cols, column);

	if (m_rows != row)
		emit rowsChanged(m_rows, row);

	qDeleteAll(m_data);
	m_data.clear();

	m_cols = column;
	m_rows = row;

	// m_data
	for (int i = 0; i < m_rows; i++) {
		QVector<qreal>* dataVec = new QVector<qreal>(m_cols);
		for (int k = 0; k < dataVec->size(); k++)
			dataVec->replace(k, qreal());
		m_data.append(dataVec);
	}
}

void TableModel::clear()
{
	qDeleteAll(m_data);
	m_data.clear();
	m_vertices.clear();

	m_rows = 0;
	m_cols = 0;

	emit layoutChanged();
}

int TableModel::randomMin()
{
	return m_randomMin;
}

int TableModel::randomMax()
{
	return m_randomMax;
}

void TableModel::setHorizontalHeaderData(const QVariantList& data)
{
	if (m_horHeaderData != data) {
		m_horHeaderData = data;
		emit layoutChanged();
	}
}

QVariantList TableModel::horizontalHeaderData() const
{
	return m_horHeaderData;
}

void TableModel::setVerticalHeaderData(const QVariantList& data)
{
	if (m_verHeaderData != data) {
		m_verHeaderData = data;
		emit layoutChanged();
	}
}

QVariantList TableModel::verticalHeaderData() const
{
	return m_verHeaderData;
}

int TableModel::rowCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent)
	return m_rows;
}

int TableModel::columnCount(const QModelIndex &parent) const
{
	Q_UNUSED(parent)
	return m_cols;
}

QVariant TableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (role != Qt::DisplayRole)
		return QVariant();

	if (orientation == Qt::Horizontal) {
		if (!m_horHeaderData.isEmpty() && m_horHeaderData.size() > section)
			return m_horHeaderData.at(section);
		else
			return QString("%1").arg(section + 1);
	}

	if (orientation == Qt::Vertical) {
		if (!m_verHeaderData.isEmpty() && m_verHeaderData.size() > section)
			return m_verHeaderData.at(section);
		else
			return QString("%1").arg(section + 1);
	}
	return QVariant();
}

QVariant TableModel::data(const QModelIndex &index, int role) const
{
	if (role == Qt::DisplayRole) {
		return m_data[index.row()]->at(index.column());
	} else if (role == Qt::EditRole) {
		return m_data[index.row()]->at(index.column());
	}

	return QVariant();
}

qreal TableModel::data(int row, int col) const
{
	return m_data[row]->at(col);
}

QVector<Tuple3ui> TableModel::verticesData() const
{
	return m_vertices;
}

bool TableModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
	if (index.isValid() && role == Qt::EditRole) {
		m_data[index.row()]->replace(index.column(), value.toDouble());
		emit dataChanged(index, index);
		return true;
	}
	return false;
}

bool TableModel::setData(int row, int column, const QVariant& value)
{
	QModelIndex index = this->createIndex(row, column);
	return setData(index, value);
}

Qt::ItemFlags TableModel::flags(const QModelIndex &index) const
{
	return QAbstractItemModel::flags(index) | Qt::ItemIsEditable;
}

} // namespace Utils
