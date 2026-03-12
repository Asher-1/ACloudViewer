// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVXMLParser.h
 * @brief XML parser producing vtkPVXMLElement tree
 */

#include "qVTK.h"             // needed for export macro
#include "vtkSmartPointer.h"  // needed for vtkSmartPointer.
#include "vtkXMLParser.h"

class vtkPVXMLElement;

/**
 * @class vtkPVXMLParser
 * @brief Parses XML into vtkPVXMLElement hierarchy
 */
class QVTK_ENGINE_LIB_API vtkPVXMLParser : public vtkXMLParser {
public:
    vtkTypeMacro(vtkPVXMLParser, vtkXMLParser);
    void PrintSelf(ostream& os, vtkIndent indent) override;
    static vtkPVXMLParser* New();

    /**
     * Write the parsed XML into the output stream.
     * @param os Output stream
     */
    void PrintXML(ostream& os);

    /**
     * Get the root element from the XML document.
     * @return Root element or nullptr if not parsed
     */
    vtkPVXMLElement* GetRootElement();

    //@{
    /**
     * If on, then the Parse method will NOT report an error using
     * vtkErrorMacro. Rather, it will just return false.  This feature is useful
     * when simply checking to see if a file is a valid XML file or there is
     * otherwise a way to recover from the failed parse.  This flag is off by
     * default.
     */
    vtkGetMacro(SuppressErrorMessages, int);
    vtkSetMacro(SuppressErrorMessages, int);
    vtkBooleanMacro(SuppressErrorMessages, int);
    //@}

    /**
     * Convenience method to parse XML contents.
     * @param xmlcontents XML string to parse
     * @param suppress_errors If true, do not report parse errors
     * @return Root element or NULL if parse failed
     */
    static vtkSmartPointer<vtkPVXMLElement> ParseXML(
            const char* xmlcontents, bool suppress_errors = false);

protected:
    vtkPVXMLParser();
    ~vtkPVXMLParser() override;

    int SuppressErrorMessages;

    void StartElement(const char* name, const char** atts) override;
    void EndElement(const char* name) override;
    void CharacterDataHandler(const char* data, int length) override;

    void AddElement(vtkPVXMLElement* element);
    void PushOpenElement(vtkPVXMLElement* element);
    vtkPVXMLElement* PopOpenElement();

    // The root XML element.
    vtkPVXMLElement* RootElement;

    // The stack of elements currently being parsed.
    vtkPVXMLElement** OpenElements;
    unsigned int NumberOfOpenElements;
    unsigned int OpenElementsSize;

    // Counter to assign unique element ids to those that don't have any.
    unsigned int ElementIdIndex;

    // Called by Parse() to read the stream and call ParseBuffer.  Can
    // be replaced by subclasses to change how input is read.
    int ParseXML() override;

    // Overridden to implement the SuppressErrorMessages feature.
    void ReportXmlParseError() override;

private:
    vtkPVXMLParser(const vtkPVXMLParser&) = delete;
    void operator=(const vtkPVXMLParser&) = delete;
};
