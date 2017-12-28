package org.apache.spark.ml.odkl

import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.Metadata

/**
  * Created by alexander.lutsenko on 20.09.16.
  */
class MetadataParam(parent: String, name: String, doc: String, isValid: Metadata => Boolean)
  extends Param[Metadata](parent, name, doc, isValid) {

  def this(parent: Identifiable, name: String, doc: String, isValid: Metadata => Boolean) =
    this(parent.uid, name, doc, isValid)

  def this(parent: String, name: String, doc: String) =
    this(parent, name, doc, (_: Metadata) => true)

  def this(parent: Identifiable, name: String, doc: String) = this(parent.uid, name, doc)

  override def jsonEncode(value: Metadata): String = {
    value.json
  }

  override def jsonDecode(json: String): Metadata = {
    Metadata.fromJson(json)
  }
}