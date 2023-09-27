# Copyright (c) 2023 Jonathan S. Pollack (https://github.com/JPPhoto)

from typing import Optional

import numpy
from PIL import Image
from pydantic import BaseModel
from scipy.ndimage import gaussian_filter1d

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.app.util.misc import SEED_MAX, get_random_seed


@invocation_output("thresholding_output")
class ThresholdingOutput(BaseInvocationOutput):
    """Thresholding output class"""

    highlights_mask: ImageField = OutputField(default=None)
    midtones_mask: ImageField = OutputField(default=None)
    shadows_mask: ImageField = OutputField(default=None)


@invocation("thresholding", title="Thresholding", tags=["thresholding"], version="1.0.0")
class ThresholdingInvocation(BaseInvocation):
    """Puts out 3 masks for a source image representing highlights, midtones, and shadows"""

    image: ImageField = InputField(description="The image to add film grain to", default=None)
    highlights_point: int = InputField(ge=0, le=255, description="Highlight point", default=170)
    shadows_point: int = InputField(ge=0, le=255, description="Shadow point", default=85)
    lut_blur: float = InputField(ge=0, description="LUT blur", default=0.0)

    def gaussian_blur(self, data):
        if self.lut_blur == 0.0:
            return data
        else:
            arr = numpy.asarray(data)
            filtered_data = gaussian_filter1d(input=arr, sigma=self.lut_blur)
            return filtered_data.tolist()

    def invoke(self, context: InvocationContext) -> ThresholdingOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        image = image.convert("L")

        highlights_lut = [0 if p > self.highlights_point else 255 for p in range(0, 256)]
        midtones_lut = [0 if (p <= self.highlights_point and p > self.shadows_point) else 255 for p in range(0, 256)]
        shadows_lut = [0 if p <= self.shadows_point else 255 for p in range(0, 256)]

        highlights_lut = self.gaussian_blur(highlights_lut)
        midtones_lut = self.gaussian_blur(midtones_lut)
        shadows_lut = self.gaussian_blur(shadows_lut)

        highlights_mask = image.point(lambda p: highlights_lut[p])
        midtones_mask = image.point(lambda p: midtones_lut[p])
        shadows_mask = image.point(lambda p: shadows_lut[p])

        h_image_dto = context.services.images.create(
            image=highlights_mask,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=None,
            workflow=self.workflow,
        )

        m_image_dto = context.services.images.create(
            image=midtones_mask,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=None,
            workflow=self.workflow,
        )

        s_image_dto = context.services.images.create(
            image=shadows_mask,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=None,
            workflow=self.workflow,
        )

        highlights_output = ImageField(image_name=h_image_dto.image_name)

        midtones_output = ImageField(image_name=m_image_dto.image_name)

        shadows_output = ImageField(image_name=s_image_dto.image_name)

        return ThresholdingOutput(
            highlights_mask=highlights_output,
            midtones_mask=midtones_output,
            shadows_mask=shadows_output,
        )
