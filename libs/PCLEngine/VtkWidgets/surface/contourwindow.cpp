#include "contourwindow.h"
#include "ui_contourwindow.h"

#include <VtkUtils/tablemodel.h>
#include <VtkUtils/modeltovectorsconverter.h>
#include <VtkUtils/pointsreader.h>
#include <VtkUtils/pointstomodelconverter.h>

#include <VtkUtils/contour.h>

#include <QThreadPool>
#include <QDebug>
#include <QFileDialog>
#include <QFileInfo>

ContourWindow::ContourWindow(QWidget *parent) :
    QWidget(parent),
    m_ui(new Ui::ContourWindow)
{
    setWindowTitle(tr("Contour"));
    m_ui->setupUi(this);
    m_ui->tableView->setAlternatingRowColors(true);

    m_model = new VtkUtils::TableModel(4, 100);
    m_ui->rowsSpinbox->setValue(100);
    m_model->setHorizontalHeaderData(QVariantList() << "X" << "Y" << "Z" << "V");
    m_model->random();

    m_contour = new VtkUtils::Contour(this);
    m_ui->verticalLayout->addWidget(m_contour);

    m_ui->tableView->setModel(m_model);

    fireupModelToVectorsConverter();
    connect(m_model, SIGNAL(dataChanged(QModelIndex,QModelIndex)), this, SLOT(onModelDataChanged(QModelIndex,QModelIndex)));
}

ContourWindow::~ContourWindow()
{
    delete m_ui;
}

void ContourWindow::onModelToVectorsConverterFinished()
{
    qDebug() << "setting new vectors";

	VtkUtils::ModelToVectorsConverter* converter = qobject_cast<VtkUtils::ModelToVectorsConverter*>(sender());
    QList<VtkUtils::Vector4F> vectors = converter->vectors();

    if (vectors.isEmpty())
        return;

    m_contour->setVectors(vectors);

    converter->deleteLater();
}

void ContourWindow::onModelDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight)
{
    Q_UNUSED(topLeft)
    Q_UNUSED(bottomRight)
    fireupModelToVectorsConverter();
}

void ContourWindow::onPointsReaderFinished()
{
	VtkUtils::PointsReader* reader = qobject_cast<VtkUtils::PointsReader*>(sender());
    QList<VtkUtils::Point3F> points = reader->points();
    if (points.isEmpty())
        return;

    QList<VtkUtils::Vector4F> vectors;
    foreach (auto pot, points)
        vectors.append(VtkUtils::Vector4F{pot.x, pot.y, pot.z, 0.0});

    m_contour->setVectors(vectors);
    reader->deleteLater();

	VtkUtils::PointsToModelConverter* converter = new VtkUtils::PointsToModelConverter(points, m_model);
    converter->setAutoDelete(false);
    connect(converter, SIGNAL(finished()), this, SLOT(onPointsToModelConverterFinished()));
    QThreadPool::globalInstance()->start(converter);
}

void ContourWindow::onPointsToModelConverterFinished()
{
	VtkUtils::PointsToModelConverter* converter = qobject_cast<VtkUtils::PointsToModelConverter*>(sender());
    emit m_model->layoutChanged();
    converter->deleteLater();
}

void ContourWindow::on_importDataButton_clicked()
{
    static QString previousDir = ".";

    QString file = QFileDialog::getOpenFileName(this, tr("Import Data"), previousDir, "All Files(*.*)");
    if (file.isEmpty())
        return;

    QFileInfo fi(file);
    previousDir = fi.canonicalPath();
    m_ui->fileEdit->setText(file);

	VtkUtils::PointsReader* reader = new VtkUtils::PointsReader(file);
    reader->setAutoDelete(false);
    connect(reader, SIGNAL(finished()), this, SLOT(onPointsReaderFinished()));
    QThreadPool::globalInstance()->start(reader);
}

void ContourWindow::on_pushButton_clicked()
{
    m_model->random();
    fireupModelToVectorsConverter();
}

void ContourWindow::on_rowsSpinbox_valueChanged(int arg1)
{
    m_model->resize(4, arg1);
    m_model->random();
    fireupModelToVectorsConverter();
}

void ContourWindow::fireupModelToVectorsConverter()
{
	VtkUtils::ModelToVectorsConverter* converter = new VtkUtils::ModelToVectorsConverter(m_model);
    connect(converter, SIGNAL(finished()), this, SLOT(onModelToVectorsConverterFinished()));
    converter->setAutoDelete(false);
    QThreadPool::globalInstance()->start(converter);
}

