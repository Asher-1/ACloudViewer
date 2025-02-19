//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

// Qt
#include <QFrame>

class QComboBox;
class QToolButton;

//! Advanced editor for color scales
/** Combo-box + shortcut to color scale editor
 **/
class ecvTextureFileSelector : public QFrame {
    Q_OBJECT

public:
    //! Default constructor
    ecvTextureFileSelector(QWidget* parent,
                           QString defaultButtonIconPath = QString());

    //! Inits selector with the Empty texture file path
    void init(const QMap<QString, QString>& texturePathMap);

    bool isEmpty() const;

    void addItem(const QString& textureFilename,
                 const QString& textureFilepath);

    //! Sets selected combo box item (scale) by UUID
    void setSelectedTexturefile(QString textureFilepath);

    QString getTexturefilePath(int index) const;

signals:

    //! Signal emitted when a texture file item is selected
    void textureFileSelected(int);

    //! Signal emitted when the user clicks on the 'Texture file loading editor'
    //! button
    void textureFileEditorSummoned();

protected:
    //! Color scales combo-box
    QComboBox* m_comboBox;

    //! Spawn color scale editor button
    QToolButton* m_button;
};