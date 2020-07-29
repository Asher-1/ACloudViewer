/*! \file
*  \brief Picture UI
*  \author Asher
*  \date 2013
*  \version 1.0
*  \copyright 2013 PERAGlobal Ltd. All rights reserved.
*
*  PenStyleBox
*/
/****************************************************************************
**
** Copyright (c) 2013 PERAGlobal Ltd. All rights reserved.
** All rights reserved.
**
****************************************************************************/

#include "penstylebox.h"

#include <algorithm>

namespace Widgets
{

	const Qt::PenStyle PenStyleBox::patterns[] = {
		Qt::SolidLine,
		Qt::DashLine,
		Qt::DotLine,
		Qt::DashDotLine,
		Qt::DashDotDotLine
	};

	/*!
	 * \brief construct PenStyleBox.
	 * \param parent, window.
	 */
	PenStyleBox::PenStyleBox(QWidget *parent) : QComboBox(parent)
	{
		setEditable(false);
		addItem("_____");
		addItem("_ _ _");
		addItem(".....");
		addItem("_._._");
		addItem("_.._..");
	}

	/*!
	 * \brief set style.
	 * \param style.
	 */
	void PenStyleBox::setStyle(const Qt::PenStyle& style)
	{
		const Qt::PenStyle*ite = std::find(patterns, 
			patterns + sizeof(patterns), style);
		if (ite == patterns + sizeof(patterns))
			this->setCurrentIndex(0);
		else
			this->setCurrentIndex(ite - patterns);
	}

	/*!
	 * \brief get style.
	 * \param index, the index of style.
	 * \return style, default: SolidLine.
	 */
	Qt::PenStyle PenStyleBox::penStyle(int index)
	{
		if (index < (int)sizeof(patterns))
			return patterns[index];
		else
			return Qt::SolidLine;
	}

	/*!
	 * \brief obtain selected style.
	 * \return style, default: SolidLine.
	 */
	Qt::PenStyle PenStyleBox::style() const
	{
		size_t i = this->currentIndex();
		if (i < sizeof(patterns))
			return patterns[i];
		else
			return Qt::SolidLine;
	}

	/*!
	 * \brief get the index of style.
	 * \param style.
	 * \return the index of style.
	 */
	int PenStyleBox::styleIndex(const Qt::PenStyle& style)
	{
		const Qt::PenStyle*ite = std::find(patterns, 
			patterns + sizeof(patterns), style);
		if (ite == patterns + sizeof(patterns))
			return 0;
		else
			return (ite - patterns);
	}
}
